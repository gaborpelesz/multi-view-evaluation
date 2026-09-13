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

#include <limits>

#include "completeness.h"

#include <pcl/common/transforms.h>
#include <pcl/io/ply_io.h>
#include <pcl/search/kdtree.h>

const int kGridCount = 2;
const float kGridShifts[kGridCount][3] = {{0.f, 0.f, 0.f}, {0.5f, 0.5f, 0.5f}};

// Completeness results for one differently-shifted voxel grid.
//
// This used to be std::unordered_map<std::tuple<int, int, int>,
// CompletenessCell>, where every cell was a separately malloc'd hash node whose
// CompletenessCell in turn owned a std::vector<size_t> with a second malloc of
// its own. Here the cell coordinates are resolved to a dense cell index by an
// open-addressing table (VoxelCellIndexMap) and the counters live in two flat
// arrays, so a cell costs no allocation of its own and a tally is an indexed
// store into contiguous memory instead of a pointer chase.
//
// The counts themselves, and which cell each point lands in, are unchanged.
struct CompletenessCellGrid {
  inline void Init(size_t tolerances_count) {
    tolerances_count_ = tolerances_count;
  }

  // Returns the dense index of the cell with the given coordinates, creating it
  // with zeroed counters if it does not exist yet.
  inline uint32_t GetCell(const VoxelCellKey& key) {
    bool inserted;
    const uint32_t cell_index = map_.Lookup(key, &inserted);
    if (inserted) {
      point_count.push_back(0);
      complete_count.resize(complete_count.size() + tolerances_count_, 0);
    }
    return cell_index;
  }

  inline size_t cell_count() const { return map_.size(); }

  // Number of scan points within a cell.
  // Indexed by: [cell_index].
  std::vector<size_t> point_count;

  // Number of complete scan points (smaller or equal to point_count), for each
  // tolerance value.
  // Indexed by: [cell_index * tolerances_count + tolerance_index].
  std::vector<size_t> complete_count;

 private:
  VoxelCellIndexMap map_;
  size_t tolerances_count_ = 0;
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

  pcl::PointXYZ search_point;
  pcl::search::KdTree<pcl::PointXYZ> reconstruction_kdtree;
  // Get sorted results from radius search. True should be the default, but be
  // on the safe side for the case of changing defaults:
  reconstruction_kdtree.setSortedResults(true);
  reconstruction_kdtree.setInputCloud(reconstruction);

  // Differently shifted voxel grids.
  // Indexed by: [map_index], then by the dense cell index that the grid assigns
  // to CalcCellCoordinates(...).
  CompletenessCellGrid cell_maps[kGridCount];
  for (int grid_index = 0; grid_index < kGridCount; ++grid_index) {
    cell_maps[grid_index].Init(tolerances_count);
  }

  const int kNN = 1;
  std::vector<int> knn_indices(kNN);
  std::vector<float> knn_squared_dists(kNN);

  // Squared search radius, reproducing bit-for-bit the threshold that
  // pcl::KdTreeFLANN::radiusSearch() used to hand to FLANN: it takes the radius
  // as a double and passes static_cast<float>(radius * radius). Promoting the
  // float tolerance to double and multiplying there is exact (a 24x24-bit
  // product fits in 53 bits), so the single rounding back to float yields
  // exactly the same value the radius search compared against.
  const float maximum_tolerance_squared =
      static_cast<float>(static_cast<double>(maximum_tolerance) *
                         static_cast<double>(maximum_tolerance));

  const long long int scan_point_size =
      static_cast<long long int>(scan->size());

// Loop over all scan points.
// knn_indices / knn_squared_dists are firstprivate rather than private so that
// each thread starts with a copy that is already sized kNN; private would
// default-construct empty vectors that the search has to grow on first use.
#pragma omp parallel for private(search_point) \
    firstprivate(knn_indices, knn_squared_dists)
  for (long long int scan_point_index = 0; scan_point_index < scan_point_size;
       ++scan_point_index) {
    const pcl::PointXYZ& scan_point = scan->at(scan_point_index);

    // Find the voxels for this scan point and increase their point count.
    // Dense cell indices are remembered rather than pointers: the flat counter
    // arrays may be reallocated by an insertion from another iteration, while
    // the index of an existing cell never changes.
    uint32_t cell_indices[kGridCount];
#pragma omp critical
    for (int grid_index = 0; grid_index < kGridCount; ++grid_index) {
      const uint32_t cell_index =
          cell_maps[grid_index].GetCell(CalcCellCoordinates(
              scan_point, voxel_size_inv, kGridShifts[grid_index][0],
              kGridShifts[grid_index][1], kGridShifts[grid_index][2]));
      ++cell_maps[grid_index].point_count[cell_index];
      cell_indices[grid_index] = cell_index;
    }

    // Find the closest reconstruction point to this scan point.
    //
    // This used to be radiusSearch(maximum_tolerance, ..., max_nn = kNN), which
    // asks the kd-tree a much more expensive question than the answer requires.
    // Only knn_squared_dists[0] is ever read below; knn_indices is never used.
    // With max_nn = 1 FLANN answers a radius query with a KNNRadiusResultSet of
    // capacity 1, i.e. it returns the single nearest neighbour, and reports it
    // only when its squared distance is strictly smaller than the squared
    // radius (KNNRadiusResultSet::addPoint returns early on dist >= worst_dist_
    // and worst_dist_ starts at radius^2; the leaf loop in
    // KDTreeSingleIndex::searchLevel likewise tests dist < worst_dist). A plain
    // 1-nearest-neighbour query returns the same nearest neighbour -- both
    // searches are exact and L2_Simple computes the squared distance
    // identically -- so testing that distance against maximum_tolerance_squared
    // with a strict '<' reproduces the radius search exactly, while letting the
    // tree shrink its search bound from the first candidate on instead of
    // descending every node that overlaps a 0.5 m ball.
    search_point.getVector3fMap() = scan_point.getVector3fMap();
    // pcl::KdTreeFLANN::nearestKSearch returns min(k, cloud size) unconditionally
    // rather than the number of neighbours actually written, so its return value
    // cannot be used to detect "nothing found" (which FLANN does produce for a
    // non-finite query point). Pre-seeding the slot with infinity makes that case
    // fail the tolerance test, which is exactly what radiusSearch returning 0 did.
    knn_squared_dists[0] = std::numeric_limits<float>::infinity();
    reconstruction_kdtree.nearestKSearch(search_point, kNN, knn_indices,
                                         knn_squared_dists);
    if (knn_squared_dists[0] < maximum_tolerance_squared) {
      int smallest_complete_tolerance_index = 0;
#pragma omp critical
      {
        // Since a reconstruction point was found within the search radius, this
        // scan point is complete for the maximum tolerance, at least.
        for (int grid_index = 0; grid_index < kGridCount; ++grid_index) {
          cell_maps[grid_index]
              .complete_count[cell_indices[grid_index] * tolerances_count +
                              tolerances_count - 1] += 1;
        }

        // Next, find the smallest tolerance for which it is still complete.
        for (int tolerance_index = tolerances_count - 2; tolerance_index >= 0;
             --tolerance_index) {
          if (sorted_tolerances_squared[tolerance_index] <
              knn_squared_dists[0]) {
            // The scan point is not completed for the current tolerance index.
            smallest_complete_tolerance_index = tolerance_index + 1;
            break;
          }
          for (int grid_index = 0; grid_index < kGridCount; ++grid_index) {
            cell_maps[grid_index]
                .complete_count[cell_indices[grid_index] * tolerances_count +
                                tolerance_index] += 1;
          }
        }
      }

      // Output point results, if requested.
      if (output_point_results) {
        // The points is incomplete for tolerances smaller than the smallest
        // complete one.
        for (int tolerance_index = 0;
             tolerance_index < smallest_complete_tolerance_index;
             ++tolerance_index) {
          point_is_complete->at(tolerance_index)[scan_point_index] = false;
        }
        // The point is complete for tolerances starting from the smallest
        // complete one.
        for (size_t tolerance_index = smallest_complete_tolerance_index;
             tolerance_index < tolerances_count; ++tolerance_index) {
          point_is_complete->at(tolerance_index)[scan_point_index] = true;
        }
      }
    } else if (output_point_results) {
      // This scan point is incomplete for all tolerances.
      for (size_t tolerance_index = 0; tolerance_index < tolerances_count;
           ++tolerance_index) {
        point_is_complete->at(tolerance_index)[scan_point_index] = false;
      }
    }
  }

  // Average results over all cells and fill the results vector.
  std::vector<double> completeness_sum(tolerances_count, 0.0);
  size_t cell_count = 0;
  for (int grid_index = 0; grid_index < kGridCount; ++grid_index) {
    const CompletenessCellGrid& grid = cell_maps[grid_index];
    const size_t grid_cell_count = grid.cell_count();
    cell_count += grid_cell_count;

    for (size_t cell_index = 0; cell_index < grid_cell_count; ++cell_index) {
      const size_t cell_point_count = grid.point_count[cell_index];
      const size_t* cell_complete_count =
          &grid.complete_count[cell_index * tolerances_count];
      for (size_t tolerance_index = 0; tolerance_index < tolerances_count;
           ++tolerance_index) {
        completeness_sum[tolerance_index] +=
            cell_complete_count[tolerance_index] / (1.0 * cell_point_count);
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
