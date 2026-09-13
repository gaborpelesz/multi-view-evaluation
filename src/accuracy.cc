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

#include "accuracy.h"

#include <algorithm>
#include <cstdint>

#include <Eigen/StdVector>
#include <pcl/io/ply_io.h>

// Number of cells used for the spatial access structure in inclination and
// azimuth directions.
const int kCellCountInclination = 1024;
const int kCellCountAzimuth = 2 * kCellCountInclination;

const int kGridCount = 2;
const float kGridShifts[kGridCount][3] = {{0.f, 0.f, 0.f}, {0.5f, 0.5f, 0.5f}};

// At least for Eigen::Vector3f, the aligned_allocator should not be necessary,
// but on the other hand it also shouldn't significantly hurt, and helps in not
// forgetting it.
typedef std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>>
    Matrix3fVector;
typedef std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>
    Vector3fVector;

// Accuracy tallies for one differently-shifted voxel grid.
//
// This used to be std::unordered_map<std::tuple<int, int, int>,
// std::vector<AccuracyCell>>, where every cell was a separately malloc'd hash
// node holding a std::vector with a second malloc of its own. Here the cell
// coordinates are resolved to a dense cell id by an open-addressing table
// (VoxelCellIndexMap) and the per-cell tallies live in flat arrays addressed by
// that id, so a cell costs no allocation of its own and a tally is an indexed
// store into contiguous memory instead of a pointer chase.
//
// The ids are handed out in first-touch order over the reconstruction points,
// so walking the flat arrays from 0 upwards visits the cells in exactly the
// order in which the original implementation created them. That is what keeps
// the floating point summation at the end of ComputeAccuracy in its original
// order.
//
// The tallies are histograms of the first tolerance index for which a point is
// accurate rather than per-tolerance counts: that is one increment per point
// and grid instead of one per tolerance, and it is what lets the classification
// loop run in parallel with a single atomic per grid. The per-tolerance
// accurate and inaccurate counts the original code kept are formed from the
// histograms afterwards and are identical to them.
struct AccuracyCellGrid {
  // Returns the dense id of the cell with the given coordinates, creating it if
  // it does not exist yet. Only ever called from the serial pass, which is what
  // makes the dense ids deterministic.
  inline uint32_t GetCell(const VoxelCellKey& key) {
    bool inserted;
    return map_.Lookup(key, &inserted);
  }

  inline size_t cell_count() const { return map_.size(); }

  // Histogram of the first tolerance index for which a reconstruction point of
  // this cell is accurate; bin tolerances_count collects the points which are
  // accurate for no tolerance at all. Sized once, after the serial pass has
  // seen every cell, so that the parallel pass can increment it in place
  // without ever reallocating.
  // Indexed by: [cell_id * (tolerances_count + 1) + tolerance_index].
  std::vector<uint32_t> accurate_histogram;

  // The same histogram, restricted to the points which additionally produced
  // inaccurate classifications.
  std::vector<uint32_t> inaccurate_histogram;

 private:
  VoxelCellIndexMap map_;
};

// Modulo which works properly for negative k (in contrast to C++' % operator),
// for example, mod(-2, 10) returns 8 instead of -2.
inline int mod(int k, int n) { return ((k %= n) < 0) ? k + n : k; }

// Represents a point in spherical coordinates.
struct SphericalPoint {
  inline SphericalPoint(const Eigen::Vector3f& p) {
    radius = p.norm();
    // See: https://en.wikipedia.org/wiki/Spherical_coordinate_system
    // For the ground truth scans, X and Y are in the horizontal plane, while
    // z points up. The coordinate system is right-handed.
    azimuth = atan2(p.y(), p.x());
    inclination = acos(p.z() / radius);
  }

  float radius;
  // From -M_PI / 2 to M_PI / 2.
  float inclination;
  // From -M_PI to M_PI.
  float azimuth;
};

// Represents a point in spherical coordinates, and in addition to
// SphericalPoint also stores the normalized direction vector to the point.
struct SphericalPointAndDirection {
  inline SphericalPointAndDirection() {}
  inline SphericalPointAndDirection(const Eigen::Vector3f& p) : point(p) {
    radius = p.norm();
    // See: https://en.wikipedia.org/wiki/Spherical_coordinate_system
    // For the ground truth scans, X and Y are in the horizontal plane, while
    // z points up. The coordinate system is right-handed.
    azimuth = atan2(p.y(), p.x());
    inclination = acos(p.z() / radius);

    // Division by radius would be faster, but let's use the special-case
    // handling for zero length that normalized() presumably has.
    direction = p.normalized();
  }

  Eigen::Vector3f point;
  Eigen::Vector3f direction;

  float radius;
  // From -M_PI / 2 to M_PI / 2.
  float inclination;
  // From -M_PI to M_PI.
  float azimuth;
};

// The subset of SphericalPointAndDirection that ClassifyPoint actually reads,
// stored by value inside the grid (see SphericalPointGrid). Keeping only these
// 28 bytes per scan point, contiguously and in cell order, means the beam test
// walks a cell's points sequentially with no indirection.
struct GridScanPoint {
  // Read for every candidate point (the beam test starts with this dot
  // product), therefore placed first.
  Eigen::Vector3f direction;
  // Read only for points that fall inside the beam volume.
  Eigen::Vector3f point;
  float radius;
};

// Stores the scan points in a grid defined on azimuth and inclination for fast
// direction-based access.
//
// The storage is a CSR / bucketed layout: one contiguous array of points in
// cell order plus an offsets array of cell_count + 1 entries, so cell c owns
// points_[cell_offsets_[c] .. cell_offsets_[c + 1]). This replaces a
// std::vector<std::vector<SphericalPointAndDirection*>>, which cost a 24-byte
// vector header for each of the 2,097,152 cells (~50 MB per scan before a
// single point is stored), one heap allocation per non-empty cell, and two
// pointer chases per visited point.
class SphericalPointGrid {
  friend class SphericalPointGridIterator;

 public:
  inline SphericalPointGrid(int cell_count_azimuth, int cell_count_inclination)
      : cell_count_azimuth_(cell_count_azimuth),
        cell_extent_azimuth_(2 * M_PI / cell_count_azimuth_),
        cell_count_inclination_(cell_count_inclination),
        cell_extent_inclination_(M_PI / cell_count_inclination) {
    cell_offsets_.assign(
        static_cast<size_t>(cell_count_azimuth_) * cell_count_inclination_ + 1,
        0u);
  }

  inline void CellCoordinatesWithoutWrap(float azimuth, float inclination,
                                         int* cell_index_azimuth,
                                         int* cell_index_inclination) const {
    float float_cell_index_azimuth = azimuth / cell_extent_azimuth_;
    *cell_index_azimuth = static_cast<int>(float_cell_index_azimuth) +
                          ((float_cell_index_azimuth < 0.f) ? (-1) : (0));

    int float_cell_index_inclination = inclination / cell_extent_inclination_;
    *cell_index_inclination =
        static_cast<int>(float_cell_index_inclination) +
        ((float_cell_index_inclination < 0.f) ? (-1) : (0));
    // Clamp.
    *cell_index_inclination = std::max(
        0, std::min(cell_count_inclination_ - 1, *cell_index_inclination));
  }

  inline int CellCoordinatesToIndex(int cell_index_azimuth,
                                    int cell_index_inclination) const {
    return cell_index_azimuth + cell_count_azimuth_ * cell_index_inclination;
  }

  // Handles clamping and wrapping around internally.
  inline int CellIndex(float azimuth, float inclination) const {
    int cell_index_azimuth;
    int cell_index_inclination;
    CellCoordinatesWithoutWrap(azimuth, inclination, &cell_index_azimuth,
                               &cell_index_inclination);

    // Wrap-around.
    cell_index_azimuth = mod(cell_index_azimuth, cell_count_azimuth_);

    return CellCoordinatesToIndex(cell_index_azimuth, cell_index_inclination);
  }

  // Converts the given scan cloud to spherical coordinates and fills the grid
  // with it.
  //
  // This is a counting sort of the points by cell index. Pass 1 converts each
  // point, pass 2 histograms the cells, the histogram is prefix-summed into the
  // offsets array, and pass 3 scatters the points into the contiguous array.
  // The scatter walks the points in increasing point index and appends to each
  // cell's running cursor, so it is stable: within a cell, the points end up in
  // increasing point index order, which is exactly the order the previous
  // push_back-based construction produced. The order in which ClassifyPoint
  // visits a cell's points is therefore unchanged. Only the conversion runs in
  // parallel; the binning stays serial and in point order.
  void Build(const PointCloud& cartesian_cloud) {
    const size_t point_count = cartesian_cloud.size();
    const size_t cell_count = cell_offsets_.size() - 1;

    // Scratch storage, freed before the next scan's grid is built.
    std::vector<GridScanPoint> unsorted_points(point_count);
    std::vector<uint32_t> point_cell_indices(point_count);

    std::fill(cell_offsets_.begin(), cell_offsets_.end(), 0u);

    // Pass 1 (parallel): spherical conversion and cell index computation. The
    // conversion and the cell index computation are the unchanged original
    // code, so every point lands in the same cell with the same field values as
    // before. Each iteration writes only its own element of the two scratch
    // arrays and reduces nothing, so the result does not depend on the thread
    // schedule.
    const long long int signed_point_count =
        static_cast<long long int>(point_count);
#pragma omp parallel for schedule(static)
    for (long long int p = 0; p < signed_point_count; ++p) {
      const SphericalPointAndDirection spherical_point(
          cartesian_cloud.at(p).getVector3fMap());

      GridScanPoint& grid_point = unsorted_points[p];
      grid_point.direction = spherical_point.direction;
      grid_point.point = spherical_point.point;
      grid_point.radius = spherical_point.radius;

      point_cell_indices[p] = static_cast<uint32_t>(
          CellIndex(spherical_point.azimuth, spherical_point.inclination));
    }

    // Pass 2 (serial): the per-cell histogram. Integer counting only; it stays
    // serial because a private copy of the histogram would cost one array of
    // cell_count entries per thread.
    for (size_t p = 0; p < point_count; ++p) {
      ++cell_offsets_[point_cell_indices[p]];
    }

    // Exclusive prefix sum: cell_offsets_[c] becomes the start of cell c.
    uint32_t running_offset = 0;
    for (size_t c = 0; c < cell_count; ++c) {
      const uint32_t count = cell_offsets_[c];
      cell_offsets_[c] = running_offset;
      running_offset += count;
    }
    cell_offsets_[cell_count] = running_offset;

    // Pass 3: stable scatter. This consumes cell_offsets_[c], advancing it to
    // the end of cell c.
    points_.resize(point_count);
    for (size_t p = 0; p < point_count; ++p) {
      points_[cell_offsets_[point_cell_indices[p]]++] = unsorted_points[p];
    }

    // Shift the consumed cursors back into start-of-cell form. Afterwards
    // cell_offsets_[c] is the start of cell c again, and the final entry still
    // holds point_count (no point index can equal cell_count, so the scatter
    // never touched it).
    for (size_t c = cell_count; c > 0; --c) {
      cell_offsets_[c] = cell_offsets_[c - 1];
    }
    cell_offsets_[0] = 0;
  }

  // The contiguous point array. May be null for an empty scan, in which case
  // every cell range is empty as well.
  inline const GridScanPoint* points_data() const { return points_.data(); }
  inline uint32_t cell_start(int cell_index) const {
    return cell_offsets_[cell_index];
  }
  inline uint32_t cell_end(int cell_index) const {
    return cell_offsets_[cell_index + 1];
  }

 private:
  int cell_count_azimuth_;
  float cell_extent_azimuth_;
  int cell_count_inclination_;
  float cell_extent_inclination_;

  // CSR layout of the scan points. cell_offsets_ has cell_count + 1 entries,
  // indexed by [cell_index_azimuth + cell_count_azimuth *
  // cell_index_inclination]; the points of that cell are
  // points_[cell_offsets_[i] .. cell_offsets_[i + 1]).
  std::vector<uint32_t> cell_offsets_;
  std::vector<GridScanPoint> points_;
};

// Iterates over an azimuth-inclination range in a SphericalPointGrid.
class SphericalPointGridIterator {
 public:
  inline SphericalPointGridIterator(const SphericalPointGrid* grid,
                                    float azimuth, float inclination,
                                    float azimuth_angle,
                                    float inclination_angle)
      : grid_(grid) {
    grid_->CellCoordinatesWithoutWrap(
        azimuth - azimuth_angle, inclination - inclination_angle,
        &min_cell_index_azimuth_, &min_cell_index_inclination_);
    grid_->CellCoordinatesWithoutWrap(
        azimuth + azimuth_angle, inclination + inclination_angle,
        &max_cell_index_azimuth_, &max_cell_index_inclination_);

    current_cell_index_azimuth_ = min_cell_index_azimuth_ - 1;
    current_cell_index_inclination_ = min_cell_index_inclination_;
  }

  inline bool Next() {
    ++current_cell_index_azimuth_;
    if (current_cell_index_azimuth_ > max_cell_index_azimuth_) {
      current_cell_index_azimuth_ = min_cell_index_azimuth_;
      ++current_cell_index_inclination_;
      return current_cell_index_inclination_ <= max_cell_index_inclination_;
    }
    return true;
  }

  inline int cell_index() const {
    return grid_->CellCoordinatesToIndex(
        mod(current_cell_index_azimuth_, grid_->cell_count_azimuth_),
        current_cell_index_inclination_);
  }

  inline int min_cell_index_azimuth() const { return min_cell_index_azimuth_; }
  inline int max_cell_index_azimuth() const { return max_cell_index_azimuth_; }
  inline int min_cell_index_inclination() const {
    return min_cell_index_inclination_;
  }
  inline int max_cell_index_inclination() const {
    return max_cell_index_inclination_;
  }

 private:
  int current_cell_index_azimuth_;
  int current_cell_index_inclination_;

  int min_cell_index_azimuth_;
  int max_cell_index_azimuth_;
  int min_cell_index_inclination_;
  int max_cell_index_inclination_;

  const SphericalPointGrid* grid_;
};

// Classifies a point as accurate, inaccurate, or unobserved given a single
// scan point cloud. This function makes use of the following: If a
// reconstruction point is classified as accurate for some tolerance, it is also
// accurate for each higher tolerance value. Thus, the function only returns the
// index of the first (lowest) tolerance for which the point is accurate, or
// accuracy_tolerances_squared.size() if the point is not classified as accurate
// for any tolerance value. Similarly, if the point is classified as inaccurate
// for any tolerance value, it will also be classified as inaccurate for any
// smaller tolerance value. At the same time, it will be classified as accurate
// for any tolerance value larger than the highest tolerance which makes it
// classify as inaccurate. Thus, the function returns a flag which, if set,
// means that every tolerance value with an index smaller than
// first_accurate_tolerance_index classifies as inaccurate. If this flag is not
// set, the classification for these tolerance values is unobserved.
inline void ClassifyPoint(const Eigen::Vector3f& cartesian_reconstruction_point,
                          const SphericalPoint& spherical_reconstruction_point,
                          float radius_horizontal,
                          // Must be sorted in increasing order.
                          const std::vector<float>& accuracy_tolerances_squared,
                          float beam_start_radius,
                          float tan_beam_divergence_halfangle_rad,
                          const SphericalPointGrid& point_grid,
                          int* first_accurate_tolerance_index,
                          bool* inaccurate_classifications_exist) {
  *first_accurate_tolerance_index = accuracy_tolerances_squared.size();
  *inaccurate_classifications_exist = false;

  // Determine the beam radius at this distance from the scanner.
  float beam_radius =
      beam_start_radius +
      spherical_reconstruction_point.radius * tan_beam_divergence_halfangle_rad;
  float beam_radius_squared = beam_radius * beam_radius;

  // Compute the bounding box of the 2D region in (azimuth, inclination) in
  // which the scan points relevant for this reconstruction point could be:
  // Find the tangents to the sphere in vertical and horizontal direction.
  float relevancy_angle_vertical;
  float relevancy_angle_horizontal;
  if (beam_radius >= spherical_reconstruction_point.radius) {
    // Search everything.
    relevancy_angle_vertical = M_PI;
    relevancy_angle_horizontal = M_PI;
  } else {
    relevancy_angle_vertical =
        asin(beam_radius / spherical_reconstruction_point.radius);
    if (beam_radius >= radius_horizontal) {
      // Search everything horizontally.
      relevancy_angle_horizontal = M_PI;
    } else {
      relevancy_angle_horizontal = asin(beam_radius / radius_horizontal);
    }
  }

  // Intersect the bounding box with the grid cells in which the scan points are
  // stored, handling wrap-around of the spherical coordinates in the horizontal
  // direction.
  SphericalPointGridIterator it(
      &point_grid, spherical_reconstruction_point.azimuth,
      spherical_reconstruction_point.inclination, relevancy_angle_horizontal,
      relevancy_angle_vertical);
  const GridScanPoint* grid_points = point_grid.points_data();
  while (it.Next()) {
    const int cell_index = it.cell_index();
    const uint32_t cell_end = point_grid.cell_end(cell_index);
    for (uint32_t point_index = point_grid.cell_start(cell_index);
         point_index < cell_end; ++point_index) {
      const GridScanPoint* scan_point = grid_points + point_index;

      // Is the reconstruction point within the beam volume? (Checked by testing
      // whether the scan ray is closer than beam_radius to the reconstruction
      // point).
      float signed_distance_along_ray =
          scan_point->direction.dot(cartesian_reconstruction_point);
      if (signed_distance_along_ray < 0) {
        // Treat points on the opposite side of the scan ray as unobserved.
        continue;
      }
      Eigen::Vector3f closest_point_on_scan_ray =
          signed_distance_along_ray * scan_point->direction;
      float scan_ray_distance_squared =
          (cartesian_reconstruction_point - closest_point_on_scan_ray)
              .squaredNorm();
      if (scan_ray_distance_squared <= beam_radius_squared) {
        // Is the reconstruction point within the region for accurate
        // classification (i.e., closer to the scan point than the evaluation
        // threshold)? In this case, early exit with accurate classification.
        float distance_from_scan_point_squared =
            (scan_point->point - cartesian_reconstruction_point).squaredNorm();
        for (size_t tolerance_index = 0;
             tolerance_index < accuracy_tolerances_squared.size() &&
             static_cast<int>(tolerance_index) <
                 *first_accurate_tolerance_index;
             ++tolerance_index) {
          if (distance_from_scan_point_squared <=
              accuracy_tolerances_squared[tolerance_index]) {
            *first_accurate_tolerance_index = tolerance_index;
            if (tolerance_index == 0) {
              // Early exit.
              return;
            }
            break;
          }
        }

        // Is the reconstruction point in front of the scan point? In this case,
        // remember that inaccurate classifications may exist.
        if (signed_distance_along_ray < scan_point->radius) {
          *inaccurate_classifications_exist = true;
        }
      }
    }
  }
}

void ComputeAccuracy(
    const MeshLabMeshInfoVector& scan_infos,
    const std::vector<PointCloudPtr>& scans, const PointCloud& reconstruction,
    float voxel_size_inv,
    // Sorted by increasing tolerance.
    const std::vector<float>& sorted_tolerances, float beam_start_radius_meters,
    float tan_beam_divergence_halfangle_rad,
    // Indexed by: [tolerance_index]. Range: [0, 1].
    std::vector<float>* results,
    // Indexed by: [tolerance_index][point_index].
    std::vector<std::vector<AccuracyResult>>* point_is_accurate) {
  bool output_point_results = point_is_accurate != nullptr;

  size_t scan_count = scans.size();
  size_t tolerances_count = sorted_tolerances.size();

  // Compute squared tolerances.
  std::vector<float> sorted_tolerances_squared(tolerances_count);
  for (size_t tolerance_index = 0; tolerance_index < tolerances_count;
       ++tolerance_index) {
    sorted_tolerances_squared[tolerance_index] =
        sorted_tolerances[tolerance_index] * sorted_tolerances[tolerance_index];
  }

  // Cache scan_T_global transformations in matrix+vector form for fast
  // global-to-scan transforming.
  Matrix3fVector scans_R_global(scan_count);
  Vector3fVector scans_T_global(scan_count);
  for (size_t scan_index = 0; scan_index < scan_count; ++scan_index) {
    const Eigen::Matrix4f& global_T_mesh = scan_infos[scan_index].global_T_mesh;

    // Rotation matrix inverse is its transpose.
    scans_R_global[scan_index] = global_T_mesh.block<3, 3>(0, 0).transpose();

    // Translation of inverse transformation is -(R^(-1) * t):
    //   y = R * x + t
    //   y - t = R * x
    //   R^(-1) * (y - t) = x
    //   x = R^(-1) * y - (R^(-1) * t)
    scans_T_global[scan_index] =
        scans_R_global[scan_index] * (-1 * global_T_mesh.block<3, 1>(0, 3));
  }

  // Transform all scan points to spherical coordinates, and sort them into grid
  // cells defined on the spherical coordinates. The grid owns the points, so no
  // separate spherical point cloud is kept alongside it.
  std::vector<std::shared_ptr<SphericalPointGrid>> point_grids(scan_count);
  for (size_t scan_index = 0; scan_index < scan_count; ++scan_index) {
    point_grids[scan_index].reset(
        new SphericalPointGrid(kCellCountAzimuth, kCellCountInclination));
    point_grids[scan_index]->Build(*scans[scan_index]);
  }

  // Prepare point_is_accurate, if requested.
  if (output_point_results) {
    point_is_accurate->resize(tolerances_count);
    for (size_t tolerance_index = 0; tolerance_index < tolerances_count;
         ++tolerance_index) {
      point_is_accurate->at(tolerance_index)
          .resize(reconstruction.size(), AccuracyResult::kUnobserved);
    }
  }

  // Differently shifted voxel grids.
  // Indexed by: [map_index], then by the dense cell id that the grid assigns to
  // CalcCellCoordinates(...), then by [tolerance_index].
  AccuracyCellGrid cell_maps[kGridCount];

  // std::vector<AccuracyResult> is a byte array, so different threads writing
  // different point indices write to distinct memory locations.
  std::vector<AccuracyResult*> point_results(tolerances_count, nullptr);
  if (output_point_results) {
    for (size_t tolerance_index = 0; tolerance_index < tolerances_count;
         ++tolerance_index) {
      point_results[tolerance_index] =
          point_is_accurate->at(tolerance_index).data();
    }
  }

  const long long int reconstruction_size =
      static_cast<long long int>(reconstruction.size());

  // Pass 1 (serial): assign every reconstruction point the cell it falls into,
  // in both voxel grids, walking the points in their original order. This
  // creates exactly the cells the original implementation created, in exactly
  // the same sequence, and hands out the dense cell ids in that same
  // first-touch order, so the order of the floating point summation over the
  // cells at the end of this function is unchanged. Only integer bookkeeping
  // happens here; the expensive classification is done in pass 2, in parallel.
  //
  // This pass is also what makes the parallel pass safe without a lock: after
  // it, every cell that will ever be touched exists and has a fixed id, so the
  // flat tally arrays can be sized once and never reallocate again.
  // Indexed by: [point_index * kGridCount + grid_index].
  std::vector<uint32_t> point_cell_ids(
      static_cast<size_t>(reconstruction_size) * kGridCount);
  for (int grid_index = 0; grid_index < kGridCount; ++grid_index) {
    AccuracyCellGrid& grid = cell_maps[grid_index];
    const float shift_x = kGridShifts[grid_index][0];
    const float shift_y = kGridShifts[grid_index][1];
    const float shift_z = kGridShifts[grid_index][2];

    // Consecutive reconstruction points usually fall into the same voxel;
    // remembering the previous key saves most of the table lookups. The cell a
    // point lands in is unaffected: the memo only short-circuits a lookup that
    // would have returned the very same id.
    VoxelCellKey previous_key = {0, 0, 0};
    uint32_t previous_id = 0;
    bool have_previous = false;

    for (long long int point_index = 0; point_index < reconstruction_size;
         ++point_index) {
      const VoxelCellKey key =
          CalcCellCoordinates(reconstruction.at(point_index), voxel_size_inv,
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
      point_cell_ids[static_cast<size_t>(point_index) * kGridCount +
                     grid_index] = cell_id;
    }
  }

  // Size the per-cell histograms now that every cell is known, and take raw
  // pointers to them: the parallel pass below only ever increments existing
  // entries, so the arrays are never reallocated while it runs.
  const size_t histogram_stride = tolerances_count + 1;
  uint32_t* accurate_histogram_ptrs[kGridCount];
  uint32_t* inaccurate_histogram_ptrs[kGridCount];
  for (int grid_index = 0; grid_index < kGridCount; ++grid_index) {
    AccuracyCellGrid& grid = cell_maps[grid_index];
    grid.accurate_histogram.assign(grid.cell_count() * histogram_stride, 0);
    grid.inaccurate_histogram.assign(grid.cell_count() * histogram_stride, 0);
    accurate_histogram_ptrs[grid_index] = grid.accurate_histogram.data();
    inaccurate_histogram_ptrs[grid_index] = grid.inaccurate_histogram.data();
  }
  const uint32_t* point_cell_ids_ptr = point_cell_ids.data();

  // Pass 2 (parallel): loop over the reconstruction points. Every iteration
  // only reads the scan grids and writes integer tallies of its own cells plus
  // its own entry of the per-point output, so the loop is free of ordering
  // constraints.
#pragma omp parallel for schedule(dynamic, 512)
  for (long long int point_index = 0; point_index < reconstruction_size;
       ++point_index) {
    const pcl::PointXYZ& point = reconstruction.at(point_index);

    int aggregate_first_accurate_tolerance_index =
        static_cast<int>(sorted_tolerances_squared.size());
    bool aggregate_inaccurate_classifications_exist = false;

    // Loop over the ground truth point clouds and classify the reconstruction
    // point for each ground truth scan. Then compute an aggregate result
    // from the individual ones: If the point is accurate given at least one
    // scan, the aggregate result is also accurate. Otherwise, if the point is
    // inaccurate given at least one scan, the aggregate result is inaccurate.
    // Otherwise, it is unobserved.
    for (size_t scan_index = 0; scan_index < scan_count; ++scan_index) {
      // Transform reconstruction point into the cloud frame.
      Eigen::Vector3f cartesian_reconstruction_point =
          scans_R_global[scan_index] * point.getVector3fMap() +
          scans_T_global[scan_index];

      // Convert it to spherical coordinates.
      SphericalPoint spherical_reconstruction_point(
          cartesian_reconstruction_point);

      // Classify it.
      SphericalPointGrid* point_grid = point_grids[scan_index].get();
      int first_accurate_tolerance_index;
      bool inaccurate_classifications_exist;
      float radius_horizontal = sqrtf(cartesian_reconstruction_point.x() *
                                          cartesian_reconstruction_point.x() +
                                      cartesian_reconstruction_point.y() *
                                          cartesian_reconstruction_point.y());
      ClassifyPoint(cartesian_reconstruction_point,
                    spherical_reconstruction_point, radius_horizontal,
                    sorted_tolerances_squared, beam_start_radius_meters,
                    tan_beam_divergence_halfangle_rad, *point_grid,
                    &first_accurate_tolerance_index,
                    &inaccurate_classifications_exist);

      // Merge into aggregate result and exit early if the point is accurate for
      // all tolerances.
      aggregate_inaccurate_classifications_exist |=
          inaccurate_classifications_exist;
      if (first_accurate_tolerance_index <
          aggregate_first_accurate_tolerance_index) {
        aggregate_first_accurate_tolerance_index =
            first_accurate_tolerance_index;
        if (aggregate_first_accurate_tolerance_index == 0) {
          break;
        }
      }
    }

    // Tally the classification into both voxel grids. The point is accurate
    // for every tolerance index from aggregate_first_accurate_tolerance_index
    // upwards, and inaccurate for every smaller one if inaccurate
    // classifications exist; both are recorded as a single histogram bin and
    // accumulated into counts below.
    for (int grid_index = 0; grid_index < kGridCount; ++grid_index) {
      const size_t bin_offset =
          static_cast<size_t>(
              point_cell_ids_ptr[static_cast<size_t>(point_index) * kGridCount +
                                 grid_index]) *
              histogram_stride +
          aggregate_first_accurate_tolerance_index;
      uint32_t* accurate_bin = accurate_histogram_ptrs[grid_index] + bin_offset;
#pragma omp atomic
      ++(*accurate_bin);
      if (aggregate_inaccurate_classifications_exist) {
        uint32_t* inaccurate_bin =
            inaccurate_histogram_ptrs[grid_index] + bin_offset;
#pragma omp atomic
        ++(*inaccurate_bin);
      }
    }

    // Output point results, if requested. Entries below the first accurate
    // tolerance index stay at their initial kUnobserved value unless
    // inaccurate classifications exist.
    if (output_point_results) {
      for (int tolerance_index = aggregate_first_accurate_tolerance_index;
           tolerance_index < static_cast<int>(tolerances_count);
           ++tolerance_index) {
        point_results[tolerance_index][point_index] = AccuracyResult::kAccurate;
      }
      if (aggregate_inaccurate_classifications_exist) {
        for (int tolerance_index = aggregate_first_accurate_tolerance_index - 1;
             tolerance_index >= 0; --tolerance_index) {
          point_results[tolerance_index][point_index] =
              AccuracyResult::kInaccurate;
        }
      }
    }
  }

  // Average results over all cells and fill the results vector. The cells are
  // walked in dense id order, which is the order in which the serial pass above
  // first touched them, so the summation order is the original one.
  std::vector<double> accuracy_sum(tolerances_count, 0.0);
  std::vector<size_t> valid_cell_count(tolerances_count, 0);
  for (int grid_index = 0; grid_index < kGridCount; ++grid_index) {
    const AccuracyCellGrid& grid = cell_maps[grid_index];
    const size_t grid_cell_count = grid.cell_count();
    uint32_t* accurate_histogram = accurate_histogram_ptrs[grid_index];
    uint32_t* inaccurate_histogram = inaccurate_histogram_ptrs[grid_index];

    for (size_t cell_id = 0; cell_id < grid_cell_count; ++cell_id) {
      // Turn this cell's histograms into the per-tolerance accurate and
      // inaccurate counts, which is exactly what the original code counted:
      // accurate_count[t]   = number of points with first_accurate <= t
      // inaccurate_count[t] = number of points with inaccurate classifications
      //                       and first_accurate > t
      uint32_t* accurate_counts = accurate_histogram + cell_id * histogram_stride;
      uint32_t running_sum = 0;
      for (size_t tolerance_index = 0; tolerance_index < tolerances_count;
           ++tolerance_index) {
        running_sum += accurate_counts[tolerance_index];
        accurate_counts[tolerance_index] = running_sum;
      }

      uint32_t* inaccurate_counts =
          inaccurate_histogram + cell_id * histogram_stride;
      running_sum = 0;
      uint32_t carry = inaccurate_counts[tolerances_count];
      for (int tolerance_index = static_cast<int>(tolerances_count) - 1;
           tolerance_index >= 0; --tolerance_index) {
        running_sum += carry;
        carry = inaccurate_counts[tolerance_index];
        inaccurate_counts[tolerance_index] = running_sum;
      }

      for (size_t tolerance_index = 0; tolerance_index < tolerances_count;
           ++tolerance_index) {
        const size_t accurate_count = accurate_counts[tolerance_index];
        const size_t inaccurate_count = inaccurate_counts[tolerance_index];
        size_t valid_point_count = accurate_count + inaccurate_count;
        if (valid_point_count > 0) {
          accuracy_sum[tolerance_index] +=
              accurate_count / (1.0f * valid_point_count);
          ++valid_cell_count[tolerance_index];
        }
      }
    }
  }

  results->resize(tolerances_count);
  for (size_t tolerance_index = 0; tolerance_index < tolerances_count;
       ++tolerance_index) {
    float* accuracy = &results->at(tolerance_index);
    if (valid_cell_count[tolerance_index] == 0) {
      *accuracy = 0;
    } else {
      *accuracy =
          accuracy_sum[tolerance_index] / valid_cell_count[tolerance_index];
    }
  }
}

void WriteAccuracyVisualization(
    const std::string& base_path, const PointCloud& reconstruction,
    // Sorted by increasing tolerance.
    const std::vector<float>& sorted_tolerances,
    // Indexed by: [tolerance_index][point_index].
    const std::vector<std::vector<AccuracyResult>>& point_is_accurate) {
  pcl::PointCloud<pcl::PointXYZRGB> accuracy_visualization;
  accuracy_visualization.resize(reconstruction.size());

  // Set visualization point positions (once for all tolerances).
  for (size_t i = 0; i < reconstruction.size(); ++i) {
    accuracy_visualization.at(i).getVector3fMap() =
        reconstruction.at(i).getVector3fMap();
  }

  // Loop over all tolerances, set visualization point colors accordingly and
  // save the point clouds.
  for (size_t tolerance_index = 0; tolerance_index < sorted_tolerances.size();
       ++tolerance_index) {
    const std::vector<AccuracyResult>& point_is_accurate_for_tolerance =
        point_is_accurate[tolerance_index];

    for (size_t point_index = 0; point_index < accuracy_visualization.size();
         ++point_index) {
      pcl::PointXYZRGB* point = &accuracy_visualization.at(point_index);
      if (point_is_accurate_for_tolerance[point_index] ==
          AccuracyResult::kAccurate) {
        // Green: accurate points.
        point->r = 0;
        point->g = 255;
        point->b = 0;
      } else if (point_is_accurate_for_tolerance[point_index] ==
                 AccuracyResult::kInaccurate) {
        // Red: inaccurate points.
        point->r = 255;
        point->g = 0;
        point->b = 0;
      } else if (point_is_accurate_for_tolerance[point_index] ==
                 AccuracyResult::kUnobserved) {
        // Blue: unobserved points.
        point->r = 0;
        point->g = 0;
        point->b = 255;
      }
    }

    std::ostringstream file_path;
    file_path << base_path << ".tolerance_"
              << sorted_tolerances[tolerance_index] << ".ply";
    pcl::io::savePLYFileBinary(file_path.str(), accuracy_visualization);
  }
}
