// Copyright 2017 Thomas Schöps
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "nn_grid.h"

#include <omp.h>

namespace {

// Cells the index is allowed per candidate point. The dense cell array costs
// 4 bytes per cell and the coordinate arrays 12 bytes per point, so this caps
// the whole structure at 28 bytes per point WHATEVER the scene extent is. That
// is the property that makes a dense cell array safe here: a large outdoor
// scene does not blow the array up, it only gets coarser cells, which costs
// query time rather than memory.
constexpr double kCellsPerPoint = 4.0;

// The shell expansion runs until the shell boundary passes the maximum
// tolerance, i.e. for at most ceil(maximum_tolerance / cell_size) + 1 shells.
// Refusing to go finer than maximum_tolerance / 8 bounds that at ten shells,
// so a query with no reconstruction point anywhere near it costs at most a
// 19^3 walk over (mostly empty) cells rather than an unbounded one.
constexpr double kMinCellsPerTolerance = 8.0;

// Largest magnitude a scaled grid coordinate may reach. Below 2^52 every
// quantity the index computes in double -- the scaled coordinate itself, its
// floor, and the difference between a coordinate and the grid origin -- is
// exact, which is what makes "a point stored in cell i has its coordinate
// inside cell i's box" an exact invariant instead of an approximate one, and
// that invariant is what licenses the pruning.
constexpr double kMaxGridCoordinate = 4.0e15;

// Coarsest grid the cell-size search below will consider. A cell 2^160 m
// across is twenty orders of magnitude wider than the widest cloud of finite
// float coordinates can be (|x| <= 3.403e38, so an extent of at most
// ~6.8e38 m), so at this exponent every axis has one or two cells and the cell
// budget is met by any cloud with at least two candidate points. The bound
// therefore exists only so the loop cannot spin: reaching it means one of the
// assumptions above is false, and Build then reports failure rather than
// returning an index whose cell ids have silently wrapped.
constexpr int kMinGridExponent = -160;

// Largest candidate count the CSR layout can address. cell_start_ holds uint32
// offsets into the point array, so the candidate set has to fit in the uint32
// domain; 2^32 - 1 points is 51 GB of coordinates, far past what the machine
// could hold, but the bound is asserted rather than assumed because violating
// it would wrap an offset instead of failing.
constexpr double kMaxCandidateCount = 4.0e9;

// Sentinel cell id for a point that is not a search candidate.
constexpr uint32_t kExcludedPoint = 0xFFFFFFFFu;

inline bool IsFinitePoint(const pcl::PointXYZ& point) {
  return std::isfinite(point.x) && std::isfinite(point.y) &&
         std::isfinite(point.z);
}

}  // namespace

bool ReconNnGrid::Build(const PointCloud& cloud, float maximum_tolerance,
                        float maximum_tolerance_squared) {
  cap_squared_ = static_cast<double>(maximum_tolerance_squared);

  cell_start_.clear();
  point_xyz_.clear();
  origin_[0] = origin_[1] = origin_[2] = 0.0;
  cell_size_ = 1.0;
  inv_cell_size_ = 1.0;
  dim_[0] = dim_[1] = dim_[2] = 1;
  total_cell_count_ = 0;

  const size_t point_count = cloud.size();
  const long long int signed_point_count =
      static_cast<long long int>(point_count);

  // Bounding box over the candidate points, which are the finite ones -- the
  // same set pcl::KdTreeFLANN::convertCloudToArray keeps. min and max are exact
  // and order-independent, so the box does not depend on the thread count.
  float min_x = std::numeric_limits<float>::infinity();
  float min_y = min_x;
  float min_z = min_x;
  float max_x = -min_x;
  float max_y = -min_x;
  float max_z = -min_x;
  long long int finite_count = 0;
#pragma omp parallel for schedule(static) reduction(min : min_x, min_y, min_z) \
    reduction(max : max_x, max_y, max_z) reduction(+ : finite_count)
  for (long long int i = 0; i < signed_point_count; ++i) {
    const pcl::PointXYZ& point = cloud.at(i);
    if (!IsFinitePoint(point)) {
      continue;
    }
    ++finite_count;
    min_x = std::min(min_x, point.x);
    min_y = std::min(min_y, point.y);
    min_z = std::min(min_z, point.z);
    max_x = std::max(max_x, point.x);
    max_y = std::max(max_y, point.y);
    max_z = std::max(max_z, point.z);
  }

  if (finite_count == 0) {
    // No candidate at all. Every query reports "nothing found", which is what
    // FLANN does with an empty index too. That is a valid index, not a
    // failure.
    return true;
  }
  if (static_cast<double>(finite_count) > kMaxCandidateCount) {
    return false;
  }

  const double lower[3] = {static_cast<double>(min_x),
                           static_cast<double>(min_y),
                           static_cast<double>(min_z)};
  const double upper[3] = {static_cast<double>(max_x),
                           static_cast<double>(max_y),
                           static_cast<double>(max_z)};

  // Pick the cell size as the finest power of two that stays inside the cell
  // budget. Powers of two only, and an origin snapped to a multiple of the cell
  // size, so that scaling a coordinate into grid space is exact -- see
  // kMaxGridCoordinate.
  const double cell_budget =
      std::min(kCellsPerPoint * static_cast<double>(finite_count),
               2.0e9);  // uint32 cell ids, with room to spare.

  double largest_magnitude = 1.0;
  for (int axis = 0; axis < 3; ++axis) {
    largest_magnitude =
        std::max(largest_magnitude,
                 std::max(std::fabs(lower[axis]), std::fabs(upper[axis])));
  }
  // The finest grid the exactness guard permits. largest_magnitude is at least
  // one, so this is at most 51 and needs no upper clamp; the tolerance guard
  // below and the budget search after it can only lower it.
  int exponent = static_cast<int>(
      std::floor(std::log2(kMaxGridCoordinate / largest_magnitude)));
  if (maximum_tolerance > 0.f && std::isfinite(maximum_tolerance)) {
    exponent = std::min(
        exponent,
        static_cast<int>(std::floor(std::log2(
            kMinCellsPerTolerance / static_cast<double>(maximum_tolerance)))));
  }
  // There is deliberately NO lower clamp on the exponent. Clamping it upwards
  // would hand back a grid FINER than the exactness guard permits, and the
  // guard is the whole reason "a point stored in cell i lies inside cell i's
  // box" is exact rather than approximate -- which is what licenses the
  // pruning. Coarsening, which is all the loop below does, only shrinks the
  // scaled coordinates, so it can never weaken the guard.

  double origin[3];
  int64_t dim[3] = {1, 1, 1};
  bool grid_fits = false;
  for (; exponent >= kMinGridExponent; --exponent) {
    const double cell_size = std::ldexp(1.0, -exponent);
    const double inv_cell_size = std::ldexp(1.0, exponent);
    // The axis dimensions stay in double until the budget test has passed.
    // A double product cannot overflow or trap however absurd the scene's
    // extent is, whereas converting an axis dimension to int64 is undefined
    // once the value leaves int64's range -- and it does leave it, for a cloud
    // whose coordinates reach 1e21 m at the exponent this search starts from.
    double cell_dim[3];
    double cells = 1.0;
    for (int axis = 0; axis < 3; ++axis) {
      origin[axis] = std::floor(lower[axis] * inv_cell_size) * cell_size;
      cell_dim[axis] =
          std::floor((upper[axis] - origin[axis]) * inv_cell_size) + 1.0;
      cells *= cell_dim[axis];
    }
    if (cells > cell_budget) {
      continue;
    }
    // Past this point the budget is met, and the budget is at most 2e9. That
    // single fact is what puts every integer the index computes in domain:
    // each axis dimension and their product fit in int64 with room to spare,
    // the flat cell id computed in pass 1 fits in uint32 without truncation,
    // and cell_start_ is an allocation of at most 8 GB rather than an
    // unbounded one. The old loop could exit WITHOUT the budget being met, and
    // then all three of those silently failed.
    for (int axis = 0; axis < 3; ++axis) {
      dim[axis] = static_cast<int64_t>(cell_dim[axis]);
    }
    cell_size_ = cell_size;
    inv_cell_size_ = inv_cell_size;
    grid_fits = true;
    break;
  }
  if (!grid_fits) {
    // Unreachable for any cloud of finite float coordinates (see
    // kMinGridExponent), and a refusal rather than an assertion so that a
    // violated assumption costs the caller the FLANN path, not a wrong number.
    return false;
  }
  for (int axis = 0; axis < 3; ++axis) {
    origin_[axis] = origin[axis];
    dim_[axis] = dim[axis];
  }
  total_cell_count_ = static_cast<uint64_t>(dim_[0]) *
                      static_cast<uint64_t>(dim_[1]) *
                      static_cast<uint64_t>(dim_[2]);

  // Counting sort of the candidate points by cell id. All three passes are
  // parallel; none of them reduces anything in floating point, and the result
  // is the same permutation at any thread count (pass 3 walks the points in
  // increasing index and appends, so it is stable).
  //
  // Pass 1: the cell id of every point.
  std::vector<uint32_t> point_cell(point_count);
#pragma omp parallel for schedule(static)
  for (long long int i = 0; i < signed_point_count; ++i) {
    const pcl::PointXYZ& point = cloud.at(i);
    if (!IsFinitePoint(point)) {
      point_cell[i] = kExcludedPoint;
      continue;
    }
    int64_t cell[3];
    const double world[3] = {static_cast<double>(point.x),
                             static_cast<double>(point.y),
                             static_cast<double>(point.z)};
    for (int axis = 0; axis < 3; ++axis) {
      cell[axis] = static_cast<int64_t>(
          std::floor(GridCoordinate(world[axis], origin_[axis],
                                    inv_cell_size_)));
      // A no-op given the exactness argument above (the origin is the floor of
      // the minimum and the extent defines the dimension), kept so that a scene
      // that somehow violated it would produce a wrong answer rather than an
      // out-of-bounds write.
      cell[axis] = std::min<int64_t>(std::max<int64_t>(cell[axis], 0),
                                     dim_[axis] - 1);
    }
    point_cell[i] = static_cast<uint32_t>((cell[2] * dim_[1] + cell[1]) *
                                              dim_[0] +
                                          cell[0]);
  }

  cell_start_.assign(static_cast<size_t>(total_cell_count_) + 1, 0u);
  point_xyz_.resize(3 * static_cast<size_t>(finite_count));

  // Per-thread block boundaries over the CELL range, not over the point range:
  // a per-thread histogram of the whole cell range would cost one array of
  // total_cell_count_ entries per thread (hundreds of megabytes), whereas
  // splitting the cells means every thread writes only inside its own slice of
  // the single shared histogram and needs no merge step at all. The price is
  // that each thread reads the cell-id array once per pass, which is sequential
  // and shared, so it comes out of cache rather than off the bus.
  std::vector<uint32_t> block_offsets;
#pragma omp parallel
  {
    const int thread_index = omp_get_thread_num();
    const int thread_count = omp_get_num_threads();
#pragma omp single
    { block_offsets.assign(static_cast<size_t>(thread_count) + 1, 0u); }
    // Implicit barrier at the end of the single: block_offsets is now visible.

    const uint64_t block_begin = total_cell_count_ *
                                 static_cast<uint64_t>(thread_index) /
                                 static_cast<uint64_t>(thread_count);
    const uint64_t block_end = total_cell_count_ *
                               static_cast<uint64_t>(thread_index + 1) /
                               static_cast<uint64_t>(thread_count);

    // Pass 2: histogram this thread's own cell block.
    uint32_t block_total = 0;
    for (size_t i = 0; i < point_count; ++i) {
      const uint32_t cell = point_cell[i];
      if (cell >= block_begin && cell < block_end) {
        ++cell_start_[cell];
        ++block_total;
      }
    }
    block_offsets[thread_index + 1] = block_total;
#pragma omp barrier

    // Exclusive prefix sum over the blocks, then over the cells of each block.
    // Tiny and serial across blocks, linear and parallel inside them.
#pragma omp single
    {
      for (size_t t = 1; t < block_offsets.size(); ++t) {
        block_offsets[t] += block_offsets[t - 1];
      }
    }
    const uint32_t block_base = block_offsets[thread_index];
    uint32_t running_offset = block_base;
    for (uint64_t cell = block_begin; cell < block_end; ++cell) {
      const uint32_t count = cell_start_[cell];
      cell_start_[cell] = running_offset;
      running_offset += count;
    }
#pragma omp barrier

    // Pass 3: stable scatter. This consumes cell_start_[c], advancing it to the
    // end of cell c.
    for (size_t i = 0; i < point_count; ++i) {
      const uint32_t cell = point_cell[i];
      if (cell >= block_begin && cell < block_end) {
        const uint32_t destination = cell_start_[cell]++;
        const pcl::PointXYZ& point = cloud.at(i);
        point_xyz_[3 * destination] = point.x;
        point_xyz_[3 * destination + 1] = point.y;
        point_xyz_[3 * destination + 2] = point.z;
      }
    }
#pragma omp barrier

    // Shift the consumed cursors back into start-of-cell form, inside each
    // block. cell_start_[c] currently holds the end of cell c, which is the
    // start of cell c + 1; the start of the block's first cell is the block
    // base this thread saved before the scatter. The last entry of the block is
    // the start of the next block's first cell, which that thread has saved as
    // its own base, so overwriting it here loses nothing.
    for (uint64_t cell = block_end; cell > block_begin + 1; --cell) {
      cell_start_[cell - 1] = cell_start_[cell - 2];
    }
    if (block_end > block_begin) {
      cell_start_[block_begin] = block_base;
    }
  }
  cell_start_[static_cast<size_t>(total_cell_count_)] =
      static_cast<uint32_t>(finite_count);
  return true;
}
