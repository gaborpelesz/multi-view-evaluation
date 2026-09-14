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

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include "completeness.h"

#include <pcl/common/transforms.h>
#include <pcl/io/ply_io.h>
#include <pcl/search/kdtree.h>

#include "nn_grid.h"

const int kGridCount = 2;
const float kGridShifts[kGridCount][3] = {{0.f, 0.f, 0.f}, {0.5f, 0.5f, 0.5f}};

// Completeness tallies for one differently-shifted voxel grid.
//
// This used to be std::unordered_map<std::tuple<int, int, int>,
// CompletenessCell>, where every cell was a separately malloc'd hash node whose
// CompletenessCell in turn owned a std::vector<size_t> with a second malloc of
// its own. Here the cell coordinates are resolved to a dense cell id by an
// open-addressing table (VoxelCellIndexMap) and the counters live in flat
// arrays addressed by that id, so a cell costs no allocation of its own and a
// tally is an indexed store into contiguous memory instead of a pointer chase.
//
// The ids are handed out in first-touch order over the scan points, so walking
// the flat arrays from 0 upwards visits the cells in exactly the order in which
// the original implementation created them. That is what keeps the floating
// point summation at the end of ComputeCompleteness in its original order.
//
// The counts themselves, and which cell each point lands in, are unchanged.
struct CompletenessCellGrid {
  // Returns the dense id of the cell with the given coordinates, creating it
  // with a zeroed point count if it does not exist yet. Only ever called from
  // the serial pass, which is what makes the dense ids deterministic.
  inline uint32_t GetCell(const VoxelCellKey& key) {
    bool inserted;
    const uint32_t cell_id = map_.Lookup(key, &inserted);
    if (inserted) {
      point_count.push_back(0);
    }
    return cell_id;
  }

  // Asks the hardware to fetch the table slot this key would probe first. A
  // pure performance hint with no effect on what GetCell later returns.
  inline void Prefetch(const VoxelCellKey& key) const { map_.Prefetch(key); }

  // Sizes the cell table and the per-cell point count array for an expected
  // number of cells, before the serial pass starts inserting. Only the table
  // geometry changes; the ids handed out do not depend on it.
  inline void Reserve(size_t expected_cells) {
    map_.Reserve(expected_cells);
    point_count.reserve(expected_cells);
  }

  inline size_t cell_count() const { return map_.size(); }

  // Number of scan points within a cell.
  // Indexed by: [cell_id].
  std::vector<size_t> point_count;

  // Histogram of the smallest tolerance index for which a scan point of the
  // cell is complete, turned into the cumulative number of complete scan points
  // per tolerance (smaller or equal to point_count) once the parallel pass is
  // done. Sized once, after the serial pass has seen every cell, so that the
  // parallel pass can increment it in place without ever reallocating.
  // Indexed by: [cell_id * tolerances_count + tolerance_index].
  std::vector<uint32_t> complete_count;

 private:
  VoxelCellIndexMap map_;
};

// The completeness classification of one squared nearest-neighbour distance:
// the smallest tolerance index for which the scan point counts as complete, or
// tolerances_count if it counts as complete for no tolerance at all.
//
// This is the ONLY channel through which a nearest-neighbour distance reaches
// the output, which is what makes the two index implementations comparable: an
// answer that differs in its last bits is invisible unless the two answers fall
// on opposite sides of one of the tolerance thresholds. --nn_verify compares
// this function's result, not just the raw floats.
//
// The test against the maximum tolerance is written as !(d < t) rather than
// (d >= t) so that a NaN distance -- which is what a non-finite scan point
// produces -- takes the "not complete" branch, exactly as the original
// if/else did.
static inline uint32_t ClassifyNnDistance(
    float nn_squared_distance, float maximum_tolerance_squared,
    const float* sorted_tolerances_squared, size_t tolerances_count) {
  if (!(nn_squared_distance < maximum_tolerance_squared)) {
    return static_cast<uint32_t>(tolerances_count);
  }
  // A reconstruction point was found within the maximum tolerance, so this scan
  // point is complete for the maximum tolerance at least. Find the smallest
  // tolerance for which it is still complete.
  int smallest_complete_tolerance_index = 0;
  for (int tolerance_index = static_cast<int>(tolerances_count) - 2;
       tolerance_index >= 0; --tolerance_index) {
    if (sorted_tolerances_squared[tolerance_index] < nn_squared_distance) {
      // The scan point is not completed for the current tolerance index.
      smallest_complete_tolerance_index = tolerance_index + 1;
      break;
    }
  }
  return static_cast<uint32_t>(smallest_complete_tolerance_index);
}

// Queries BOTH nearest neighbour indices over every scan point and reports how
// often they disagree; aborts if any disagreement could change a point's
// classification.
//
// This exists because exactly one link in the bit-exactness argument for
// ReconNnGrid is not a proof but a statement about code generation: that the
// compiler turns the three-term float accumulation in the grid's inner loop
// into the same rounding sequence that the precompiled FLANN inside
// libpcl_kdtree uses (on arm64, an fmadd chain). That is true on the hosts this
// was developed and gated on, and it is checkable at runtime anywhere -- so it
// is checked here rather than assumed, over the real scene and all of its
// points, which is a far stronger test than a synthetic self-check over a few
// thousand points would be.
//
// It is deliberately opt-in: running it always would build the kd-tree and
// double the query work in the very phase this index exists to make fast, which
// would corrupt the measurement. Run it once per host and per scene.
static void VerifyNnIndices(const PointCloud& scan, const ReconNnGrid& grid,
                            pcl::search::KdTree<pcl::PointXYZ>& kdtree,
                            float maximum_tolerance_squared,
                            const float* sorted_tolerances_squared,
                            size_t tolerances_count) {
  const long long int scan_point_size =
      static_cast<long long int>(scan.size());
  long long int value_mismatches = 0;
  long long int band_mismatches = 0;

#pragma omp parallel reduction(+ : value_mismatches, band_mismatches)
  {
    pcl::PointXYZ search_point;
    pcl::Indices knn_indices(1);
    std::vector<float> knn_squared_dists(1);

#pragma omp for schedule(dynamic, 4096)
    for (long long int scan_point_index = 0; scan_point_index < scan_point_size;
         ++scan_point_index) {
      const pcl::PointXYZ& scan_point = scan.at(scan_point_index);
      const float grid_squared_distance = grid.NearestSquaredDistance(
          scan_point.x, scan_point.y, scan_point.z);

      search_point.getVector3fMap() = scan_point.getVector3fMap();
      knn_squared_dists[0] = std::numeric_limits<float>::infinity();
      kdtree.nearestKSearch(search_point, 1, knn_indices, knn_squared_dists);
      const float flann_squared_distance = knn_squared_dists[0];

      // Two answers that are both at or above the maximum tolerance are
      // interchangeable by construction -- that is the whole point of the
      // grid's search radius cap -- so only their bits are compared when at
      // least one of them is below it.
      if ((grid_squared_distance < maximum_tolerance_squared ||
           flann_squared_distance < maximum_tolerance_squared) &&
          std::memcmp(&grid_squared_distance, &flann_squared_distance,
                      sizeof(float)) != 0) {
        ++value_mismatches;
      }
      if (ClassifyNnDistance(grid_squared_distance, maximum_tolerance_squared,
                             sorted_tolerances_squared, tolerances_count) !=
          ClassifyNnDistance(flann_squared_distance, maximum_tolerance_squared,
                             sorted_tolerances_squared, tolerances_count)) {
        ++band_mismatches;
      }
    }
  }

  std::fprintf(stderr,
               "nn_verify: grid cell size %g m over %llu cells, %zu candidate "
               "points\n",
               grid.cell_size(),
               static_cast<unsigned long long>(grid.cell_count()),
               grid.point_count());
  std::fprintf(stderr,
               "nn_verify: %lld scan points, %lld squared distances differing "
               "in any bit, %lld differing classifications\n",
               scan_point_size, value_mismatches, band_mismatches);
  if (band_mismatches != 0) {
    std::fprintf(stderr,
                 "nn_verify: FAILED -- the grid index does not reproduce the "
                 "kd-tree's classification on this host. Re-run with "
                 "--nn_index flann.\n");
    std::abort();
  }
}

void ComputeCompleteness(const MeshLabMeshInfoVector& scan_infos,
                         const std::vector<PointCloudPtr>& scans,
                         const PointCloudPtr& reconstruction,
                         float voxel_size_inv,
                         // Sorted by increasing tolerance.
                         const std::vector<float>& sorted_tolerances,
                         NnIndexKind nn_index, bool nn_verify,
                         // Indexed by: [tolerance_index]. Range: [0, 1].
                         std::vector<float>* results,
                         // Indexed by: [tolerance_index][scan_point_index].
                         std::vector<std::vector<bool>>* point_is_complete) {
  bool output_point_results = point_is_complete != nullptr;
  size_t tolerances_count = sorted_tolerances.size();
  float maximum_tolerance = sorted_tolerances.back();

  // Compute squared tolerances.
  std::vector<float> sorted_tolerances_squared(tolerances_count);
  for (size_t tolerance_index = 0; tolerance_index < tolerances_count;
       ++tolerance_index) {
    sorted_tolerances_squared[tolerance_index] =
        sorted_tolerances[tolerance_index] * sorted_tolerances[tolerance_index];
  }

  // Merge the scan point clouds into one point cloud in global coordinates.
  PointCloudPtr scan(new PointCloud());
  for (size_t scan_index = 0; scan_index < scan_infos.size(); ++scan_index) {
    PointCloud temp_cloud;
    pcl::transformPointCloud(*scans[scan_index], temp_cloud,
                             scan_infos[scan_index].global_T_mesh);
    *scan += temp_cloud;
  }

  // Prepare point_is_complete, if requested.
  if (output_point_results) {
    point_is_complete->resize(tolerances_count);
    for (size_t tolerance_index = 0; tolerance_index < tolerances_count;
         ++tolerance_index) {
      point_is_complete->at(tolerance_index).resize(scan->size(), false);
    }
  }

  // Special case for empty reconstructions since the KdTree construction
  // crashes for them.
  if (reconstruction->size() == 0) {
    results->clear();
    results->resize(tolerances_count, 0.f);
    return;
  }

  // Differently shifted voxel grids.
  // Indexed by: [map_index], then by the dense cell id that the grid assigns to
  // CalcCellCoordinates(...).
  CompletenessCellGrid cell_maps[kGridCount];

  const int kNN = 1;

  // Squared search radius, reproducing bit-for-bit the threshold that
  // pcl::KdTreeFLANN::radiusSearch() used to hand to FLANN: it takes the radius
  // as a double and passes static_cast<float>(radius * radius). Promoting the
  // float tolerance to double and multiplying there is exact (a 24x24-bit
  // product fits in 53 bits), so the single rounding back to float yields
  // exactly the same value the radius search compared against.
  const float maximum_tolerance_squared =
      static_cast<float>(static_cast<double>(maximum_tolerance) *
                         static_cast<double>(maximum_tolerance));

  // The nearest neighbour index. ReconNnGrid is a uniform-voxel CSR index built
  // by a parallel counting sort; pcl::search::KdTree is the original FLANN
  // kd-tree, whose build is a single-threaded half second and which is kept
  // reachable as the reference implementation. Only one of the two is built
  // unless --nn_verify asked for both.
  bool use_grid = (nn_index == NnIndexKind::kGrid);
  bool verify_nn = nn_verify;
  ReconNnGrid reconstruction_grid;
  if (use_grid || verify_nn) {
    if (!reconstruction_grid.Build(*reconstruction, maximum_tolerance,
                                   maximum_tolerance_squared)) {
      // The grid refuses to index this cloud (its coordinates are so extreme
      // that no power-of-two grid satisfies both the exactness guard and the
      // cell budget). Say so on stderr and answer from the kd-tree, which has
      // no such bound. Falling back silently would be worse than the crash it
      // replaces: the number would still be printed, and nothing in the run
      // record would say which index produced it.
      std::fprintf(stderr,
                   "nn_index: the grid cannot represent this reconstruction "
                   "cloud (its coordinates leave the index's exact domain); "
                   "answering from the kd-tree instead, as --nn_index flann "
                   "would.\n");
      use_grid = false;
      verify_nn = false;
    }
  }
  pcl::search::KdTree<pcl::PointXYZ> reconstruction_kdtree;
  if (!use_grid || verify_nn) {
    // Get sorted results from radius search. True should be the default, but be
    // on the safe side for the case of changing defaults:
    reconstruction_kdtree.setSortedResults(true);
    reconstruction_kdtree.setInputCloud(reconstruction);
  }

  const long long int scan_point_size =
      static_cast<long long int>(scan->size());

  // Pass 1 (serial): assign every scan point the cell it falls into, in both
  // voxel grids, walking the points in their original order. This creates
  // exactly the cells the original implementation created, in exactly the same
  // sequence, and hands out the dense cell ids in that same first-touch order,
  // so the order of the floating point summation over the cells at the end of
  // this function is unchanged. Only integer bookkeeping happens here; the
  // expensive nearest neighbour search is done in pass 2, in parallel.
  //
  // This pass is also what makes the parallel pass safe without a lock: after
  // it, every cell that will ever be touched exists and has a fixed id, so the
  // flat tally arrays can be sized once and never reallocate again.
  // Indexed by: [grid_index * scan_point_size + scan_point_index]. Grid-major,
  // so that each grid's ids are one sequential write stream; the point-major
  // layout this used to have interleaved the two grids and made both streams
  // strided. The array is not read until pass 1 has finished, so only the
  // address an already-determined id is stored at changes.
  std::vector<uint32_t> point_cell_ids(
      static_cast<size_t>(scan_point_size) * kGridCount);
  // Size both cell tables and their point count arrays before anything is
  // inserted. Left to grow from its 1024-slot default, each table doubles its
  // way up to millions of slots and re-probes every cell it holds at every
  // doubling, which costs more random probes than answering the lookups does.
  // The estimate is a sample of the cell space rather than of the points, so it
  // tracks the number of cells at any density and each table -- and each point
  // count array -- ends up the size it would have grown to anyway.
  size_t expected_cells[kGridCount];
  EstimateDistinctCellCounts(scan->points.data(),
                             static_cast<size_t>(scan_point_size),
                             voxel_size_inv, kGridShifts, expected_cells);
  for (int grid_index = 0; grid_index < kGridCount; ++grid_index) {
    cell_maps[grid_index].Reserve(expected_cells[grid_index]);
  }
  // Both grids are walked together, one point at a time. kGridCount is a
  // compile-time 2, so the grid loop unrolls and the point array is streamed
  // once instead of once per grid. Every grid keeps its own table, its own id
  // stream, its own point counts and its own memo below, and never reads
  // another grid's state, so interleaving the two cannot change either grid's
  // sequence of first-touched cells -- each still sees the points in strictly
  // ascending index order.
  uint32_t* grid_cell_ids[kGridCount];
  // If a point falls into the same voxel as the previous one, remembering that
  // cell saves a table lookup; measured on the bench dataset, that is the case
  // for ~7% of the scan points (they are still in scanline order). The cell a
  // point lands in is unaffected: the memo only short-circuits a lookup that
  // would have returned the very same id.
  VoxelCellKey previous_key[kGridCount];
  uint32_t previous_id[kGridCount];
  bool have_previous[kGridCount];
  for (int grid_index = 0; grid_index < kGridCount; ++grid_index) {
    grid_cell_ids[grid_index] = point_cell_ids.data() +
                                static_cast<size_t>(grid_index) *
                                    scan_point_size;
    previous_key[grid_index] = VoxelCellKey{0, 0, 0};
    previous_id[grid_index] = 0;
    have_previous[grid_index] = false;
  }

  // The lookups themselves are random accesses into a 134 MB table and miss
  // cache almost every time, while the points are read sequentially, so the key
  // of a point some distance ahead can be computed for free and its table slot
  // fetched while the current point is being resolved. 12 points ahead measured
  // best in the 8-16 range.
  const long long int kPrefetchDistance = 12;

  for (long long int scan_point_index = 0; scan_point_index < scan_point_size;
       ++scan_point_index) {
    if (scan_point_index + kPrefetchDistance < scan_point_size) {
      const pcl::PointXYZ& ahead_point =
          scan->at(scan_point_index + kPrefetchDistance);
      for (int grid_index = 0; grid_index < kGridCount; ++grid_index) {
        cell_maps[grid_index].Prefetch(CalcCellCoordinates(
            ahead_point, voxel_size_inv, kGridShifts[grid_index][0],
            kGridShifts[grid_index][1], kGridShifts[grid_index][2]));
      }
    }

    const pcl::PointXYZ& scan_point = scan->at(scan_point_index);
    for (int grid_index = 0; grid_index < kGridCount; ++grid_index) {
      const VoxelCellKey key = CalcCellCoordinates(
          scan_point, voxel_size_inv, kGridShifts[grid_index][0],
          kGridShifts[grid_index][1], kGridShifts[grid_index][2]);
      uint32_t cell_id;
      if (have_previous[grid_index] && key == previous_key[grid_index]) {
        cell_id = previous_id[grid_index];
      } else {
        cell_id = cell_maps[grid_index].GetCell(key);
        previous_key[grid_index] = key;
        previous_id[grid_index] = cell_id;
        have_previous[grid_index] = true;
      }
      ++cell_maps[grid_index].point_count[cell_id];
      grid_cell_ids[grid_index][scan_point_index] = cell_id;
    }
  }

  // Size the per-cell histograms now that every cell is known, and take raw
  // pointers to them: the parallel pass below only ever increments existing
  // entries, so the arrays are never reallocated while it runs.
  uint32_t* histogram_ptrs[kGridCount];
  for (int grid_index = 0; grid_index < kGridCount; ++grid_index) {
    CompletenessCellGrid& grid = cell_maps[grid_index];
    grid.complete_count.assign(grid.cell_count() * tolerances_count, 0);
    histogram_ptrs[grid_index] = grid.complete_count.data();
  }
  const uint32_t* point_cell_ids_ptrs[kGridCount];
  for (int grid_index = 0; grid_index < kGridCount; ++grid_index) {
    point_cell_ids_ptrs[grid_index] =
        point_cell_ids.data() + static_cast<size_t>(grid_index) *
                                    scan_point_size;
  }

  // Smallest tolerance index for which a scan point is complete, or
  // tolerances_count if it is complete for no tolerance. Only filled in if the
  // per-point output is requested. The per-point results are collected here
  // rather than written straight into point_is_complete because
  // std::vector<bool> is a bitfield: threads writing neighbouring point indices
  // would read-modify-write the same word. This vector has one element per
  // point, so parallel writes touch distinct objects.
  std::vector<uint32_t> smallest_complete_tolerance_indices;
  if (output_point_results) {
    smallest_complete_tolerance_indices.assign(
        static_cast<size_t>(scan_point_size),
        static_cast<uint32_t>(tolerances_count));
  }

  const float* const sorted_tolerances_squared_ptr =
      sorted_tolerances_squared.data();

  if (verify_nn) {
    VerifyNnIndices(*scan, reconstruction_grid, reconstruction_kdtree,
                    maximum_tolerance_squared, sorted_tolerances_squared_ptr,
                    tolerances_count);
  }

  // Pass 2 (parallel): the nearest neighbour search. Every iteration reads the
  // index and writes only integer tallies of its own cells plus its own entry
  // of the per-point output, so the loop is free of ordering constraints.
#pragma omp parallel
  {
    // Declared inside the parallel region rather than with a private() clause:
    // private() default-constructs, which handed every thread empty vectors.
    pcl::PointXYZ search_point;
    pcl::Indices knn_indices(kNN);
    std::vector<float> knn_squared_dists(kNN);

#pragma omp for schedule(dynamic, 4096)
    for (long long int scan_point_index = 0; scan_point_index < scan_point_size;
         ++scan_point_index) {
      const pcl::PointXYZ& scan_point = scan->at(scan_point_index);

      // Find the closest reconstruction point to this scan point.
      //
      // This used to be radiusSearch(maximum_tolerance, ..., max_nn = kNN),
      // which asks the index a much more expensive question than the answer
      // requires. Only the squared distance is ever read below; the neighbour's
      // index is never used. With max_nn = 1 FLANN answers a radius query with
      // a KNNRadiusResultSet of capacity 1, i.e. it returns the single nearest
      // neighbour, and reports it only when its squared distance is strictly
      // smaller than the squared radius (KNNRadiusResultSet::addPoint returns
      // early on dist >= worst_dist_ and worst_dist_ starts at radius^2; the
      // leaf loop in KDTreeSingleIndex::searchLevel likewise tests
      // dist < worst_dist). A plain 1-nearest-neighbour query returns the same
      // nearest neighbour -- both searches are exact and L2_Simple computes the
      // squared distance identically -- so testing that distance against
      // maximum_tolerance_squared with a strict '<' reproduces the radius
      // search exactly.
      float nn_squared_distance;
      if (use_grid) {
        nn_squared_distance = reconstruction_grid.NearestSquaredDistance(
            scan_point.x, scan_point.y, scan_point.z);
      } else {
        search_point.getVector3fMap() = scan_point.getVector3fMap();
        // pcl::KdTreeFLANN::nearestKSearch returns min(k, cloud size)
        // unconditionally rather than the number of neighbours actually
        // written, so its return value cannot be used to detect "nothing found"
        // (which FLANN does produce for a non-finite query point). Pre-seeding
        // the slot with infinity makes that case fail the tolerance test, which
        // is exactly what radiusSearch returning 0 did.
        knn_squared_dists[0] = std::numeric_limits<float>::infinity();
        reconstruction_kdtree.nearestKSearch(search_point, kNN, knn_indices,
                                             knn_squared_dists);
        nn_squared_distance = knn_squared_dists[0];
      }

      const uint32_t smallest_complete_tolerance_index = ClassifyNnDistance(
          nn_squared_distance, maximum_tolerance_squared,
          sorted_tolerances_squared_ptr, tolerances_count);
      if (smallest_complete_tolerance_index < tolerances_count) {
        // The point is complete for every tolerance index from
        // smallest_complete_tolerance_index upwards. Record that as a single
        // histogram bin; the cumulative counts are formed below. Counting the
        // histogram instead of every tolerance separately needs only one atomic
        // increment per point and grid, independent of the tolerance count.
        for (int grid_index = 0; grid_index < kGridCount; ++grid_index) {
          uint32_t* bin =
              histogram_ptrs[grid_index] +
              static_cast<size_t>(
                  point_cell_ids_ptrs[grid_index][scan_point_index]) *
                  tolerances_count +
              smallest_complete_tolerance_index;
#pragma omp atomic
          ++(*bin);
        }

        if (output_point_results) {
          smallest_complete_tolerance_indices[scan_point_index] =
              smallest_complete_tolerance_index;
        }
      }
    }
  }

  // Output point results, if requested. The points are incomplete for
  // tolerances smaller than the smallest complete one and complete from there
  // on; the vectors are already initialized to false.
  if (output_point_results) {
    for (size_t tolerance_index = 0; tolerance_index < tolerances_count;
         ++tolerance_index) {
      std::vector<bool>& point_is_complete_for_tolerance =
          point_is_complete->at(tolerance_index);
      for (long long int scan_point_index = 0;
           scan_point_index < scan_point_size; ++scan_point_index) {
        if (smallest_complete_tolerance_indices[scan_point_index] <=
            tolerance_index) {
          point_is_complete_for_tolerance[scan_point_index] = true;
        }
      }
    }
  }

  // Average results over all cells and fill the results vector. The cells are
  // walked in dense id order, which is the order in which the serial pass above
  // first touched them, so the summation order is the original one.
  std::vector<double> completeness_sum(tolerances_count, 0.0);
  size_t cell_count = 0;
  for (int grid_index = 0; grid_index < kGridCount; ++grid_index) {
    const CompletenessCellGrid& grid = cell_maps[grid_index];
    const size_t grid_cell_count = grid.cell_count();
    cell_count += grid_cell_count;

    uint32_t* histogram = histogram_ptrs[grid_index];
    for (size_t cell_id = 0; cell_id < grid_cell_count; ++cell_id) {
      // Turn this cell's histogram into the cumulative number of complete
      // points per tolerance, which is exactly what the original code counted.
      uint32_t* cell_complete_count = histogram + cell_id * tolerances_count;
      uint32_t running_sum = 0;
      for (size_t tolerance_index = 0; tolerance_index < tolerances_count;
           ++tolerance_index) {
        running_sum += cell_complete_count[tolerance_index];
        cell_complete_count[tolerance_index] = running_sum;
      }

      const size_t cell_point_count = grid.point_count[cell_id];
      for (size_t tolerance_index = 0; tolerance_index < tolerances_count;
           ++tolerance_index) {
        const size_t complete_count = cell_complete_count[tolerance_index];
        completeness_sum[tolerance_index] +=
            complete_count / (1.0 * cell_point_count);
      }
    }
  }

  results->resize(tolerances_count);
  for (size_t tolerance_index = 0; tolerance_index < tolerances_count;
       ++tolerance_index) {
    results->at(tolerance_index) =
        completeness_sum[tolerance_index] / cell_count;
  }
}

void WriteCompletenessVisualization(
    const std::string& base_path, const MeshLabMeshInfoVector& scan_infos,
    const std::vector<PointCloudPtr>& scans,
    // Sorted by increasing tolerance.
    const std::vector<float>& sorted_tolerances,
    // Indexed by: [tolerance_index][scan_point_index].
    const std::vector<std::vector<bool>>& point_is_complete) {
  pcl::PointCloud<pcl::PointXYZRGB> completeness_visualization;
  completeness_visualization.resize(point_is_complete[0].size());

  // Set visualization point positions (once for all tolerances).
  size_t output_index = 0;
  for (size_t scan_index = 0; scan_index < scan_infos.size(); ++scan_index) {
    PointCloud temp_cloud;
    pcl::transformPointCloud(*scans[scan_index], temp_cloud,
                             scan_infos[scan_index].global_T_mesh);
    for (size_t i = 0; i < temp_cloud.size(); ++i) {
      completeness_visualization.at(output_index).getVector3fMap() =
          temp_cloud.at(i).getVector3fMap();
      ++output_index;
    }
  }

  // Loop over all tolerances, set visualization point colors accordingly and
  // save the point clouds.
  for (size_t tolerance_index = 0; tolerance_index < sorted_tolerances.size();
       ++tolerance_index) {
    const std::vector<bool>& point_is_complete_for_tolerance =
        point_is_complete[tolerance_index];

    for (size_t scan_point_index = 0;
         scan_point_index < completeness_visualization.size();
         ++scan_point_index) {
      pcl::PointXYZRGB* point =
          &completeness_visualization.at(scan_point_index);
      if (point_is_complete_for_tolerance[scan_point_index]) {
        // Green: complete points.
        point->r = 0;
        point->g = 255;
        point->b = 0;
      } else {
        // Red: incomplete points.
        point->r = 255;
        point->g = 0;
        point->b = 0;
      }
    }

    std::ostringstream file_path;
    file_path << base_path << ".tolerance_"
              << sorted_tolerances[tolerance_index] << ".ply";
    pcl::io::savePLYFileBinary(file_path.str(), completeness_visualization);
  }
}
