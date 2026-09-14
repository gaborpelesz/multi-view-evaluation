// Copyright 2017 Thomas Schöps, Johannes L. Schönberger
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

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "util.h"

// One-sided short circuit for the completeness nearest neighbour query.
//
// WHAT THE QUERY IS ACTUALLY FOR. ComputeCompleteness reduces the single float
// it gets from the kd-tree to an integer band: the smallest tolerance index for
// which the scan point is complete. Finding ANY reconstruction point within the
// finest tolerance t0 of a scan point pins that band at 0, its floor, because
// the true nearest neighbour can then only be closer still. For such a point
// the exact nearest neighbour distance is never needed -- it is computed,
// compared against the tolerance ladder, and thrown away.
//
// WHAT THIS IS. A dense uniform voxel grid over the reconstruction in CSR
// layout, answering exactly one question: "is there a reconstruction point
// within t0 of q?". The cell size is at least 2 * t0, so the t0-ball around a
// query touches at most a 2x2x2 block of cells, and the cell holding the query
// itself is probed first because it alone answers most queries.
//
// WHY IT CANNOT CHANGE A RESULT. The filter never replaces the kd-tree's
// answer; it only declines to ask the question when the answer is already
// forced. It accepts only when it finds a point at squared distance
// <= t0^2 * (1 - 2^-14), and then reports band 0 without inventing a distance.
// Everything else -- every query it declines, and the entire fallback path --
// runs the original code untouched. See the acceptance margin argument on
// accept_squared_ below.
class Band0VoxelFilter {
 public:
  Band0VoxelFilter()
      : enabled_(false),
        accept_squared_(0.f),
        inv_cell_size_(0.0),
        ball_cells_(0.0),
        origin_x_(0),
        origin_y_(0),
        origin_z_(0),
        count_x_(0),
        count_y_(0),
        count_z_(0) {}

  // Builds the grid over the reconstruction. finest_tolerance is
  // sorted_tolerances[0] and finest_tolerance_squared is
  // sorted_tolerances_squared[0]; the caller has already established that
  // reporting band 0 is sound for them (see ComputeCompleteness). Returns
  // whether the filter can be used; if it cannot, HasPointWithin() always
  // declines and the original code path runs for every query.
  bool Build(const PointCloud& reconstruction, float finest_tolerance,
             float finest_tolerance_squared) {
    enabled_ = false;
    if (!(finest_tolerance > 0.f) || !(finest_tolerance_squared > 0.f)) {
      return false;
    }

    // ACCEPTANCE MARGIN. Our kernel and flann::L2_Simple evaluate the same
    // three-term float expression over the same coordinates: the per-axis
    // differences and their squares are correctly rounded float operations and
    // therefore bit-identical, and the accumulation is written below in
    // L2_Simple's own left-to-right order. The only remaining degree of freedom
    // is FMA contraction, which removes at most two roundings and so moves the
    // result by well under 1e-6 relative. Accepting only at 2^-14 = 6.1e-5
    // below t0^2 leaves that discrepancy no room to matter: whenever we accept,
    // FLANN's own squared distance for the same pair is still < t0^2, so its
    // reported minimum is < t0^2 and the band walk yields 0.
    accept_squared_ =
        finest_tolerance_squared * (1.0f - 0x1p-14f);

    // The CSR offsets are uint32_t, so a cloud they could not address is
    // declined outright rather than silently wrapped. PCL cannot produce one
    // today; this is here so that it cannot start to.
    if (reconstruction.size() > 0xFFFFFFFEull) {
      return false;
    }
    const long long int point_count =
        static_cast<long long int>(reconstruction.size());
    if (point_count <= 0) {
      return false;
    }

    // Bounding box over the points the kd-tree actually indexes.
    // pcl::KdTreeFLANN::convertCloudToArray drops every point that is not
    // finite, so a candidate must be finite to be a point FLANN could have
    // found; non-finite points are skipped here for the same reason.
    float min_x = std::numeric_limits<float>::infinity();
    float min_y = min_x;
    float min_z = min_x;
    float max_x = -std::numeric_limits<float>::infinity();
    float max_y = max_x;
    float max_z = max_x;
#pragma omp parallel for schedule(static) reduction(min : min_x, min_y, min_z) \
    reduction(max : max_x, max_y, max_z)
    for (long long int i = 0; i < point_count; ++i) {
      const pcl::PointXYZ& p = reconstruction.at(i);
      if (!IsUsable(p.x, p.y, p.z)) {
        continue;
      }
      min_x = std::min(min_x, p.x);
      min_y = std::min(min_y, p.y);
      min_z = std::min(min_z, p.z);
      max_x = std::max(max_x, p.x);
      max_y = std::max(max_y, p.y);
      max_z = std::max(max_z, p.z);
    }
    if (!(min_x <= max_x)) {
      // No usable point at all.
      return false;
    }

    // Cell size: a power of two, so multiplying a coordinate by its reciprocal
    // is exact in double and a point's cell index is computed without any
    // rounding at all. At least 2 * t0, which bounds the block a query has to
    // probe at 2x2x2, and then coarsened until the dense cell array fits the
    // budget -- an ETH3D outdoor scene spans hundreds of metres and a 1/16 m
    // grid over it would be tens of gigabytes.
    double cell_size = 1.0 / 16.0;
    const double needed_cell_size = 2.0 * static_cast<double>(finest_tolerance);
    for (int i = 0; i < 128 && cell_size < needed_cell_size; ++i) {
      cell_size *= 2.0;
    }
    if (!(cell_size >= needed_cell_size)) {
      return false;
    }

    long long int origin[3] = {0, 0, 0};
    long long int counts[3] = {0, 0, 0};
    const float lower[3] = {min_x, min_y, min_z};
    const float upper[3] = {max_x, max_y, max_z};
    bool fits = false;
    for (int attempt = 0; attempt < 128; ++attempt) {
      const double inv = 1.0 / cell_size;
      double total_cells = 1.0;
      for (int axis = 0; axis < 3; ++axis) {
        const long long int lo = static_cast<long long int>(
            std::floor(static_cast<double>(lower[axis]) * inv));
        const long long int hi = static_cast<long long int>(
            std::floor(static_cast<double>(upper[axis]) * inv));
        origin[axis] = lo;
        counts[axis] = hi - lo + 1;
        total_cells *= static_cast<double>(counts[axis]);
      }
      if (total_cells <= static_cast<double>(kMaxCellCount)) {
        fits = true;
        break;
      }
      cell_size *= 2.0;
    }
    if (!fits) {
      return false;
    }

    inv_cell_size_ = 1.0 / cell_size;
    ball_cells_ = static_cast<double>(finest_tolerance) * inv_cell_size_;
    origin_x_ = origin[0];
    origin_y_ = origin[1];
    origin_z_ = origin[2];
    count_x_ = static_cast<int>(counts[0]);
    count_y_ = static_cast<int>(counts[1]);
    count_z_ = static_cast<int>(counts[2]);

    const size_t cell_count = static_cast<size_t>(count_x_) *
                              static_cast<size_t>(count_y_) *
                              static_cast<size_t>(count_z_);

    // Counting sort of the reconstruction into the cells. Unlike the two voxel
    // tallies in completeness.cc and the spherical grid in accuracy.cc, this
    // one carries no ordering obligation whatsoever: the only thing ever asked
    // of a cell is whether it holds a point within t0, so which of its points
    // comes first, and which of several qualifying points is the one that stops
    // the scan, cannot be observed in the output. That is what allows the
    // histogram and the scatter to be atomic and unordered, and hence parallel.
    cell_start_.assign(cell_count + 1, 0u);
    uint32_t* cell_start = cell_start_.data();
#pragma omp parallel for schedule(static)
    for (long long int i = 0; i < point_count; ++i) {
      const pcl::PointXYZ& p = reconstruction.at(i);
      const uint32_t cell = PointCell(p.x, p.y, p.z);
      if (cell == kNoCell) {
        continue;
      }
#pragma omp atomic
      ++cell_start[cell + 1];
    }

    // Exclusive prefix sum: cell_start_[c] becomes the first slot of cell c and
    // cell_start_[cell_count] the number of usable points.
    uint32_t running_offset = 0;
    for (size_t c = 1; c <= cell_count; ++c) {
      running_offset += cell_start[c];
      cell_start[c] = running_offset;
    }

    // Scatter. Each point claims its slot with an atomic post-increment on a
    // private copy of the offsets, so cell_start_ itself stays in
    // start-of-cell form and needs no repair pass afterwards.
    std::vector<uint32_t> cursors(cell_start_.begin(),
                                  cell_start_.begin() + cell_count);
    uint32_t* cursor = cursors.data();
    points_.assign(static_cast<size_t>(running_offset) * 4, 0.f);
    float* points = points_.data();
#pragma omp parallel for schedule(static)
    for (long long int i = 0; i < point_count; ++i) {
      const pcl::PointXYZ& p = reconstruction.at(i);
      const uint32_t cell = PointCell(p.x, p.y, p.z);
      if (cell == kNoCell) {
        continue;
      }
      uint32_t slot;
#pragma omp atomic capture
      slot = cursor[cell]++;
      float* destination = points + static_cast<size_t>(slot) * 4;
      destination[0] = p.x;
      destination[1] = p.y;
      destination[2] = p.z;
      destination[3] = 0.f;
    }

    enabled_ = true;
    return true;
  }

  inline bool enabled() const { return enabled_; }

  // True if some reconstruction point is at squared distance <=
  // accept_squared_ from (qx, qy, qz). A false answer means "not established",
  // never "there is none": the caller must then run the kd-tree query. Every
  // early exit below -- the filter being disabled, a non-finite or far-away
  // query, a cell block that misses the point that would have qualified -- is
  // therefore safe by construction.
  inline bool HasPointWithin(float qx, float qy, float qz) const {
    if (!enabled_) {
      return false;
    }
    // Rejects non-finite and absurdly distant queries in one comparison: any
    // NaN or infinity makes the sum non-finite and the test false. Everything
    // that passes has |coordinate| < 1e12, so multiplying by the reciprocal
    // cell size (at most 16) stays far below 2^52 and the floor/fraction
    // arithmetic below is exact.
    if (!(std::fabs(static_cast<double>(qx)) +
              std::fabs(static_cast<double>(qy)) +
              std::fabs(static_cast<double>(qz)) <
          kCoordinateLimit)) {
      return false;
    }

    int lo_x, hi_x, home_x;
    int lo_y, hi_y, home_y;
    int lo_z, hi_z, home_z;
    if (!AxisRange(qx, origin_x_, count_x_, &lo_x, &hi_x, &home_x) ||
        !AxisRange(qy, origin_y_, count_y_, &lo_y, &hi_y, &home_y) ||
        !AxisRange(qz, origin_z_, count_z_, &lo_z, &hi_z, &home_z)) {
      return false;
    }

    // The cell the query sits in holds the qualifying point for most queries,
    // so it is probed on its own before the surrounding block is considered.
    const uint32_t home_cell = CellIndex(home_x, home_y, home_z);
    if (ScanCell(home_cell, qx, qy, qz)) {
      return true;
    }
    for (int z = lo_z; z <= hi_z; ++z) {
      for (int y = lo_y; y <= hi_y; ++y) {
        for (int x = lo_x; x <= hi_x; ++x) {
          const uint32_t cell = CellIndex(x, y, z);
          if (cell == home_cell) {
            continue;
          }
          if (ScanCell(cell, qx, qy, qz)) {
            return true;
          }
        }
      }
    }
    return false;
  }

 private:
  static const uint32_t kNoCell = 0xFFFFFFFFu;
  // 32 M cells, i.e. 128 MB of offsets. Beyond this the grid is coarsened.
  static const uint32_t kMaxCellCount = 32u * 1024u * 1024u;
  static constexpr double kCoordinateLimit = 1e12;

  static inline bool IsUsable(float x, float y, float z) {
    return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
  }

  inline uint32_t CellIndex(int x, int y, int z) const {
    return static_cast<uint32_t>((z * count_y_ + y) * count_x_ + x);
  }

  // Cell of a reconstruction point, or kNoCell if it is not indexable. Exact:
  // inv_cell_size_ is a power of two, so the product is exact in double and the
  // floor introduces no rounding of its own.
  inline uint32_t PointCell(float x, float y, float z) const {
    if (!IsUsable(x, y, z)) {
      return kNoCell;
    }
    const long long int cx =
        static_cast<long long int>(
            std::floor(static_cast<double>(x) * inv_cell_size_)) -
        origin_x_;
    const long long int cy =
        static_cast<long long int>(
            std::floor(static_cast<double>(y) * inv_cell_size_)) -
        origin_y_;
    const long long int cz =
        static_cast<long long int>(
            std::floor(static_cast<double>(z) * inv_cell_size_)) -
        origin_z_;
    // The bounding box was taken over exactly these points, so this holds; the
    // check is here so that a future change to the box cannot silently corrupt
    // memory.
    if (cx < 0 || cx >= count_x_ || cy < 0 || cy >= count_y_ || cz < 0 ||
        cz >= count_z_) {
      return kNoCell;
    }
    return CellIndex(static_cast<int>(cx), static_cast<int>(cy),
                     static_cast<int>(cz));
  }

  // Cells of one axis that the t0-ball around q can reach, clamped to the grid,
  // plus the cell q itself sits in. Returns false if the ball misses the grid
  // on this axis entirely.
  inline bool AxisRange(float q, long long int origin, int count, int* lo,
                        int* hi, int* home) const {
    const double scaled = static_cast<double>(q) * inv_cell_size_;
    const double floored = std::floor(scaled);
    // Exact for |scaled| < 2^52, which the caller has established.
    const double fraction = scaled - floored;
    const long long int base = static_cast<long long int>(floored) - origin;

    // The ball reaches the previous cell iff it crosses this cell's lower face,
    // and the next cell iff it crosses the upper one. ball_cells_ = t0 / h is
    // at most 0.5, so at most one neighbour is added per side. A rounding error
    // in ball_cells_ could at worst drop a neighbouring cell from the block,
    // which can only cost a fall-through to the kd-tree.
    long long int low = base;
    long long int high = base;
    if (fraction < ball_cells_) {
      --low;
    }
    if (fraction + ball_cells_ >= 1.0) {
      ++high;
    }

    const long long int last = static_cast<long long int>(count) - 1;
    if (high < 0 || low > last) {
      return false;
    }
    if (low < 0) {
      low = 0;
    }
    if (high > last) {
      high = last;
    }
    long long int center = base;
    if (center < low) {
      center = low;
    } else if (center > high) {
      center = high;
    }
    *lo = static_cast<int>(low);
    *hi = static_cast<int>(high);
    *home = static_cast<int>(center);
    return true;
  }

  // Squared distance in flann::L2_Simple's own evaluation order: the per-axis
  // differences are correctly rounded float subtractions (negating the operand
  // order is exact and the square removes the sign), the squares are correctly
  // rounded float multiplications, and the sum is accumulated left to right
  // just as L2_Simple's loop accumulates it. Only FMA contraction can make this
  // differ from FLANN's value at all, and accept_squared_ leaves room for it.
  inline bool ScanCell(uint32_t cell, float qx, float qy, float qz) const {
    const uint32_t begin = cell_start_[cell];
    const uint32_t end = cell_start_[cell + 1];
    const float* point = points_.data() + static_cast<size_t>(begin) * 4;
    for (uint32_t i = begin; i < end; ++i, point += 4) {
      const float dx = qx - point[0];
      const float dy = qy - point[1];
      const float dz = qz - point[2];
      float squared_distance = dx * dx;
      squared_distance += dy * dy;
      squared_distance += dz * dz;
      if (squared_distance <= accept_squared_) {
        return true;
      }
    }
    return false;
  }

  bool enabled_;
  float accept_squared_;
  double inv_cell_size_;
  // t0 expressed in cells, i.e. t0 / cell_size. At most 0.5 by construction.
  double ball_cells_;
  long long int origin_x_;
  long long int origin_y_;
  long long int origin_z_;
  int count_x_;
  int count_y_;
  int count_z_;

  // CSR layout of the reconstruction. cell_start_ has cell_count + 1 entries;
  // the points of cell c occupy points_[4 * cell_start_[c] .. 4 *
  // cell_start_[c + 1]) as (x, y, z, padding) quadruples.
  std::vector<uint32_t> cell_start_;
  std::vector<float> points_;
};
