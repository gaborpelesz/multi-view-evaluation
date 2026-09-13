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

#include "completeness.h"

#include <cstdint>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <pcl/common/transforms.h>
#include <pcl/io/ply_io.h>
#include <pcl/search/kdtree.h>

const int kGridCount = 2;
const float kGridShifts[kGridCount][3] = {{0.f, 0.f, 0.f}, {0.5f, 0.5f, 0.5f}};

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

  // Differently shifted voxel grids.
  // Indexed by: [map_index][CalcCellCoordinates(...)]. The mapped value is a
  // dense cell id: the per-cell tallies are kept in flat arrays addressed by
  // that id instead of inside the map, so that they can be updated with atomic
  // integer increments from several threads.
  std::unordered_map<std::tuple<int, int, int>, uint32_t>
      cell_maps[kGridCount];

  const long long int scan_point_size =
      static_cast<long long int>(scan->size());

  // Pass 1 (serial): assign every scan point the cell it falls into, in both
  // voxel grids, walking the points in their original order. This creates
  // exactly the cells the original implementation created, in exactly the same
  // sequence, so the iteration order of the maps -- and with it the order of
  // the floating point summation over the cells at the end of this function --
  // is unchanged. Only integer bookkeeping happens here; the expensive nearest
  // neighbour search is done in pass 2, in parallel.
  // Indexed by: [scan_point_index * kGridCount + grid_index].
  std::vector<uint32_t> point_cell_ids(
      static_cast<size_t>(scan_point_size) * kGridCount);
  // Indexed by: [grid_index][cell_id]. Number of scan points in the cell.
  std::vector<uint32_t> cell_point_counts[kGridCount];
  for (int grid_index = 0; grid_index < kGridCount; ++grid_index) {
    auto& cell_map = cell_maps[grid_index];
    std::vector<uint32_t>& point_counts = cell_point_counts[grid_index];
    const float shift_x = kGridShifts[grid_index][0];
    const float shift_y = kGridShifts[grid_index][1];
    const float shift_z = kGridShifts[grid_index][2];

    // Consecutive scan points usually fall into the same voxel; remembering
    // the previous key saves most of the hash lookups.
    std::tuple<int, int, int> previous_key;
    uint32_t previous_id = 0;
    bool have_previous = false;

    for (long long int scan_point_index = 0; scan_point_index < scan_point_size;
         ++scan_point_index) {
      const std::tuple<int, int, int> key = CalcCellCoordinates(
          scan->at(scan_point_index), voxel_size_inv, shift_x, shift_y,
          shift_z);
      uint32_t cell_id;
      if (have_previous && key == previous_key) {
        cell_id = previous_id;
      } else {
        auto insertion = cell_map.try_emplace(
            key, static_cast<uint32_t>(point_counts.size()));
        cell_id = insertion.first->second;
        if (insertion.second) {
          point_counts.push_back(0);
        }
        previous_key = key;
        previous_id = cell_id;
        have_previous = true;
      }
      ++point_counts[cell_id];
      point_cell_ids[static_cast<size_t>(scan_point_index) * kGridCount +
                     grid_index] = cell_id;
    }
  }

  // Histogram of the smallest tolerance index for which a scan point is
  // complete, per cell. Indexed by: [grid_index][cell_id * tolerances_count +
  // smallest_complete_tolerance_index]. After the parallel pass this is turned
  // into the cumulative "number of complete points per tolerance" in place,
  // which is what the original code counted directly. Counting the histogram
  // instead needs only one atomic increment per point and grid.
  std::vector<uint32_t> complete_histograms[kGridCount];
  for (int grid_index = 0; grid_index < kGridCount; ++grid_index) {
    complete_histograms[grid_index].assign(
        cell_point_counts[grid_index].size() * tolerances_count, 0);
  }
  uint32_t* histogram_ptrs[kGridCount];
  for (int grid_index = 0; grid_index < kGridCount; ++grid_index) {
    histogram_ptrs[grid_index] = complete_histograms[grid_index].data();
  }
  const uint32_t* point_cell_ids_ptr = point_cell_ids.data();

  // Smallest tolerance index for which a scan point is complete, or
  // tolerances_count if it is complete for no tolerance. Only filled in if the
  // per-point output is requested. The per-point results are collected here
  // rather than written straight into point_is_complete because
  // std::vector<bool> is a bitfield: threads writing neighbouring point
  // indices would write the same word. This vector has one element per point,
  // so parallel writes touch distinct objects.
  std::vector<uint32_t> smallest_complete_tolerance_indices;
  if (output_point_results) {
    smallest_complete_tolerance_indices.assign(
        static_cast<size_t>(scan_point_size),
        static_cast<uint32_t>(tolerances_count));
  }

  const int kNN = 1;

  // Pass 2 (parallel): the nearest neighbour search. Every iteration reads the
  // kd-tree and writes only integer tallies of its own cells, so the loop is
  // free of ordering constraints.
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

      // Find the closest reconstruction point to this scan point, limited to
      // the maximum evaluation tolerance for efficiency.
      search_point.getVector3fMap() = scan_point.getVector3fMap();
      if (reconstruction_kdtree.radiusSearch(search_point, maximum_tolerance,
                                             knn_indices, knn_squared_dists,
                                             kNN) > 0) {
        // Since a reconstruction point was found within the search radius,
        // this scan point is complete for the maximum tolerance, at least.
        // Find the smallest tolerance for which it is still complete.
        int smallest_complete_tolerance_index = 0;
        for (int tolerance_index = static_cast<int>(tolerances_count) - 2;
             tolerance_index >= 0; --tolerance_index) {
          if (sorted_tolerances_squared[tolerance_index] <
              knn_squared_dists[0]) {
            smallest_complete_tolerance_index = tolerance_index + 1;
            break;
          }
        }

        // The point is complete for every tolerance index from
        // smallest_complete_tolerance_index upwards. Record that as a single
        // histogram bin; the cumulative counts are formed below.
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

  // Average results over all cells and fill the results vector.
  std::vector<double> completeness_sum(tolerances_count, 0.0);
  size_t cell_count = 0;
  for (int grid_index = 0; grid_index < kGridCount; ++grid_index) {
    cell_count += cell_maps[grid_index].size();

    // Turn the per-cell histogram into the cumulative number of complete
    // points per tolerance, which is exactly what the original code counted.
    uint32_t* histogram = histogram_ptrs[grid_index];
    const size_t grid_cell_count = cell_point_counts[grid_index].size();
    for (size_t cell_id = 0; cell_id < grid_cell_count; ++cell_id) {
      uint32_t* cell_histogram = histogram + cell_id * tolerances_count;
      uint32_t running_sum = 0;
      for (size_t tolerance_index = 0; tolerance_index < tolerances_count;
           ++tolerance_index) {
        running_sum += cell_histogram[tolerance_index];
        cell_histogram[tolerance_index] = running_sum;
      }
    }

    for (auto it = cell_maps[grid_index].cbegin(),
              end = cell_maps[grid_index].cend();
         it != end; ++it) {
      const uint32_t cell_id = it->second;
      const uint32_t* cell_histogram = histogram + cell_id * tolerances_count;
      const size_t point_count = cell_point_counts[grid_index][cell_id];
      for (size_t tolerance_index = 0; tolerance_index < tolerances_count;
           ++tolerance_index) {
        const size_t complete_count = cell_histogram[tolerance_index];
        completeness_sum[tolerance_index] +=
            complete_count / (1.0 * point_count);
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
