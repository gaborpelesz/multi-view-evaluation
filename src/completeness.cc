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
#include <cstdlib>
#include <limits>
#include <vector>

#include "completeness.h"

#include <pcl/common/transforms.h>
#include <pcl/io/ply_io.h>
#include <pcl/search/kdtree.h>

#include "band0_filter.h"

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

void ComputeCompleteness(const MeshLabMeshInfoVector& scan_infos,
                         const std::vector<PointCloudPtr>& scans,
                         const PointCloudPtr& reconstruction,
                         float voxel_size_inv,
                         // Sorted by increasing tolerance.
                         const std::vector<float>& sorted_tolerances,
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

  pcl::search::KdTree<pcl::PointXYZ> reconstruction_kdtree;
  // Get sorted results from radius search. True should be the default, but be
  // on the safe side for the case of changing defaults:
  reconstruction_kdtree.setSortedResults(true);
  reconstruction_kdtree.setInputCloud(reconstruction);

  // Squared search radius, reproducing bit-for-bit the threshold that
  // pcl::KdTreeFLANN::radiusSearch() used to hand to FLANN: it takes the radius
  // as a double and passes static_cast<float>(radius * radius). Promoting the
  // float tolerance to double and multiplying there is exact (a 24x24-bit
  // product fits in 53 bits), so the single rounding back to float yields
  // exactly the same value the radius search compared against.
  const float maximum_tolerance_squared =
      static_cast<float>(static_cast<double>(maximum_tolerance) *
                         static_cast<double>(maximum_tolerance));

  // Pre-filter for the nearest neighbour query below. A scan point with any
  // reconstruction point within the finest tolerance is complete for every
  // tolerance, so its band is 0 and the exact nearest neighbour distance is not
  // needed; the filter establishes that for the majority of the scan points at
  // a small fraction of the cost of a kd-tree descent, and declines for the
  // rest, which then take the original path unchanged.
  //
  // Two preconditions make reporting band 0 sound, and the filter stays off if
  // either fails. The band walk ends at 0 iff sorted_tolerances_squared[0] is
  // not below the reported squared distance, which the filter's acceptance
  // threshold guarantees, but only if that threshold is above zero at all. And
  // the outer test is a strict 'knn_squared_dists[0] < maximum_tolerance_
  // squared', which an all-zero tolerance list would fail even for a point
  // lying exactly on a reconstruction point.
  Band0VoxelFilter band0_filter;
  if (sorted_tolerances.front() > 0.f && maximum_tolerance_squared > 0.f) {
    band0_filter.Build(*reconstruction, sorted_tolerances.front(),
                       sorted_tolerances_squared.front());
  }

  // Differently shifted voxel grids.
  // Indexed by: [map_index], then by the dense cell id that the grid assigns to
  // CalcCellCoordinates(...).
  CompletenessCellGrid cell_maps[kGridCount];

  const int kNN = 1;

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
  // Indexed by: [scan_point_index * kGridCount + grid_index].
  std::vector<uint32_t> point_cell_ids(
      static_cast<size_t>(scan_point_size) * kGridCount);
  for (int grid_index = 0; grid_index < kGridCount; ++grid_index) {
    CompletenessCellGrid& grid = cell_maps[grid_index];
    const float shift_x = kGridShifts[grid_index][0];
    const float shift_y = kGridShifts[grid_index][1];
    const float shift_z = kGridShifts[grid_index][2];

    // Consecutive scan points usually fall into the same voxel; remembering
    // the previous key saves most of the table lookups. The cell a point lands
    // in is unaffected: the memo only short-circuits a lookup that would have
    // returned the very same id.
    VoxelCellKey previous_key = {0, 0, 0};
    uint32_t previous_id = 0;
    bool have_previous = false;

    for (long long int scan_point_index = 0; scan_point_index < scan_point_size;
         ++scan_point_index) {
      const VoxelCellKey key =
          CalcCellCoordinates(scan->at(scan_point_index), voxel_size_inv,
                              shift_x, shift_y, shift_z);
      uint32_t cell_id;
      if (have_previous && key == previous_key) {
        cell_id = previous_id;
      } else {
        cell_id = grid.GetCell(key);
        previous_key = key;
        previous_id = cell_id;
        have_previous = true;
      }
      ++grid.point_count[cell_id];
      point_cell_ids[static_cast<size_t>(scan_point_index) * kGridCount +
                     grid_index] = cell_id;
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
  const uint32_t* point_cell_ids_ptr = point_cell_ids.data();

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

  // Number of scan points the pre-filter answered on its own, for the
  // fallback-rate report below. Accumulated per thread in a register and folded
  // once per thread, so the hot loop pays nothing for it.
  long long int band0_accepted = 0;

  // Pass 2 (parallel): the nearest neighbour search. Every iteration reads the
  // kd-tree and writes only integer tallies of its own cells plus its own entry
  // of the per-point output, so the loop is free of ordering constraints.
#pragma omp parallel
  {
    // Declared inside the parallel region rather than with a private() clause:
    // private() default-constructs, which handed every thread empty vectors.
    pcl::PointXYZ search_point;
    pcl::Indices knn_indices(kNN);
    std::vector<float> knn_squared_dists(kNN);
    long long int thread_band0_accepted = 0;

#pragma omp for schedule(dynamic, 4096)
    for (long long int scan_point_index = 0; scan_point_index < scan_point_size;
         ++scan_point_index) {
      const pcl::PointXYZ& scan_point = scan->at(scan_point_index);

      // The pre-filter answers "there is a reconstruction point within the
      // finest tolerance", which forces the smallest complete tolerance index
      // to 0 and makes the kd-tree query unnecessary. It never answers anything
      // else: when it declines, the original code below runs unchanged, and
      // when it accepts, no distance is synthesised -- the band is written
      // directly at its floor. See Band0VoxelFilter for why an acceptance
      // cannot disagree with what the kd-tree would have reported.
      if (band0_filter.HasPointWithin(scan_point.x, scan_point.y,
                                      scan_point.z)) {
        ++thread_band0_accepted;
        for (int grid_index = 0; grid_index < kGridCount; ++grid_index) {
          uint32_t* bin =
              histogram_ptrs[grid_index] +
              static_cast<size_t>(
                  point_cell_ids_ptr[static_cast<size_t>(scan_point_index) *
                                         kGridCount +
                                     grid_index]) *
                  tolerances_count;
#pragma omp atomic
          ++(*bin);
        }

        if (output_point_results) {
          smallest_complete_tolerance_indices[scan_point_index] = 0;
        }
        continue;
      }

      // Find the closest reconstruction point to this scan point.
      //
      // This used to be radiusSearch(maximum_tolerance, ..., max_nn = kNN),
      // which asks the kd-tree a much more expensive question than the answer
      // requires. Only knn_squared_dists[0] is ever read below; knn_indices is
      // never used. With max_nn = 1 FLANN answers a radius query with a
      // KNNRadiusResultSet of capacity 1, i.e. it returns the single nearest
      // neighbour, and reports it only when its squared distance is strictly
      // smaller than the squared radius (KNNRadiusResultSet::addPoint returns
      // early on dist >= worst_dist_ and worst_dist_ starts at radius^2; the
      // leaf loop in KDTreeSingleIndex::searchLevel likewise tests
      // dist < worst_dist). A plain 1-nearest-neighbour query returns the same
      // nearest neighbour -- both searches are exact and L2_Simple computes the
      // squared distance identically -- so testing that distance against
      // maximum_tolerance_squared with a strict '<' reproduces the radius
      // search exactly, while letting the tree shrink its search bound from the
      // first candidate on instead of descending every node that overlaps a
      // 0.5 m ball.
      search_point.getVector3fMap() = scan_point.getVector3fMap();
      // pcl::KdTreeFLANN::nearestKSearch returns min(k, cloud size)
      // unconditionally rather than the number of neighbours actually written,
      // so its return value cannot be used to detect "nothing found" (which
      // FLANN does produce for a non-finite query point). Pre-seeding the slot
      // with infinity makes that case fail the tolerance test, which is exactly
      // what radiusSearch returning 0 did.
      knn_squared_dists[0] = std::numeric_limits<float>::infinity();
      reconstruction_kdtree.nearestKSearch(search_point, kNN, knn_indices,
                                           knn_squared_dists);
      if (knn_squared_dists[0] < maximum_tolerance_squared) {
        // Since a reconstruction point was found within the search radius, this
        // scan point is complete for the maximum tolerance, at least. Find the
        // smallest tolerance for which it is still complete.
        int smallest_complete_tolerance_index = 0;
        for (int tolerance_index = static_cast<int>(tolerances_count) - 2;
             tolerance_index >= 0; --tolerance_index) {
          if (sorted_tolerances_squared[tolerance_index] <
              knn_squared_dists[0]) {
            // The scan point is not completed for the current tolerance index.
            smallest_complete_tolerance_index = tolerance_index + 1;
            break;
          }
        }

        // The point is complete for every tolerance index from
        // smallest_complete_tolerance_index upwards. Record that as a single
        // histogram bin; the cumulative counts are formed below. Counting the
        // histogram instead of every tolerance separately needs only one atomic
        // increment per point and grid, independent of the tolerance count.
        for (int grid_index = 0; grid_index < kGridCount; ++grid_index) {
          uint32_t* bin =
              histogram_ptrs[grid_index] +
              static_cast<size_t>(
                  point_cell_ids_ptr[static_cast<size_t>(scan_point_index) *
                                         kGridCount +
                                     grid_index]) *
                  tolerances_count +
              smallest_complete_tolerance_index;
#pragma omp atomic
          ++(*bin);
        }

        if (output_point_results) {
          smallest_complete_tolerance_indices[scan_point_index] =
              static_cast<uint32_t>(smallest_complete_tolerance_index);
        }
      }
    }

#pragma omp atomic
    band0_accepted += thread_band0_accepted;
  }

  // The hit rate is a property of the scene, not of the program: a
  // reconstruction that covers the ground truth poorly falls back for most of
  // its scan points and the filter then only costs its build. Reporting it
  // keeps that visible in a campaign record instead of hidden in a wall clock.
  if (std::getenv("MVE_BAND0_STATS") != nullptr) {
    std::fprintf(stderr,
                 "band0 filter: %s, %lld of %lld scan points answered "
                 "(%.2f%% fell back to the kd-tree)\n",
                 band0_filter.enabled() ? "on" : "off", band0_accepted,
                 scan_point_size,
                 scan_point_size > 0
                     ? 100.0 * static_cast<double>(scan_point_size -
                                                   band0_accepted) /
                           static_cast<double>(scan_point_size)
                     : 0.0);
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
