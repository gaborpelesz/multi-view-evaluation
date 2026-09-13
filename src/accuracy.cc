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

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include <Eigen/StdVector>
#include <pcl/io/ply_io.h>

// ---------------------------------------------------------------------------
// SIMD kernel selection.
//
// The innermost loop of ComputeAccuracy() tests candidate scan points against
// the laser beam cone of a reconstruction point. It is implemented three times
// -- AVX2, NEON and portable -- with an identical structure, and all three are
// only ever used as a *conservative rejection filter*: a candidate that the
// filter does not reject is re-evaluated by EvaluateCandidateExact(), which
// contains the unmodified upstream Eigen arithmetic. The filter therefore
// cannot influence the result, only how much work is skipped. See
// ComputeBeamFilter() for the error bound that makes the rejection safe.
// ---------------------------------------------------------------------------
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define ETH3D_CLASSIFY_LANES 8
#define ETH3D_CLASSIFY_ISA "avx2"
#elif defined(__ARM_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#define ETH3D_CLASSIFY_LANES 4
#define ETH3D_CLASSIFY_ISA "neon"
#else
#define ETH3D_CLASSIFY_LANES 4
#define ETH3D_CLASSIFY_ISA "portable"
#endif

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

// Accuracy results for one voxel cell.
struct AccuracyCell {
  inline AccuracyCell() : accurate_count(0), inaccurate_count(0) {}

  // Number of accurate reconstruction points within this cell.
  size_t accurate_count;

  // Number of inaccurate reconstruction points within this cell.
  size_t inaccurate_count;
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
// Used while building ScanPointGrid; the grid stores the resulting floats in
// its own layout and does not keep these objects.
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

// Stores the scan points of one ground truth scan in a grid defined on azimuth
// and inclination for fast direction-based access.
//
// Layout: the points are stored once, sorted by grid cell, in two flat arrays;
// cell_start_ gives the half-open index range of each cell. This replaces the
// upstream std::vector<std::vector<SphericalPointAndDirection*>>, which needed
// a 24-byte std::vector header per cell (50 MB for the 2,097,152 cells of one
// scan, touched at random) plus a pointer indirection into a separate
// array-of-structs for every candidate point. Here a cell lookup touches two
// adjacent uint32 (8.4 MB per scan), and the candidates of a whole azimuth run
// are contiguous, which is what makes the blocked SIMD filter possible at all.
//
//   directions_[3 * i + {0,1,2}]  normalized direction to scan point i
//   points_[4 * i + {0,1,2}]      position of scan point i
//   points_[4 * i + 3]            distance of scan point i from the scanner
//
// directions_ is kept separate from points_ because the filter only reads
// directions; positions and radii are read for the (few) candidates that
// survive the filter.
class ScanPointGrid {
 public:
  inline ScanPointGrid(int cell_count_azimuth, int cell_count_inclination)
      : cell_count_azimuth_(cell_count_azimuth),
        cell_extent_azimuth_(2 * M_PI / cell_count_azimuth_),
        cell_count_inclination_(cell_count_inclination),
        cell_extent_inclination_(M_PI / cell_count_inclination),
        cell_count_(static_cast<size_t>(cell_count_azimuth) *
                    static_cast<size_t>(cell_count_inclination)) {}

  // Bins the given scan into the grid. Byte-identical to the upstream binning:
  // the same SphericalPointAndDirection values are computed and the same
  // CellIndex() decides the cell; only the storage layout differs.
  void Build(const PointCloud& cloud) {
    const size_t point_count = cloud.size();

    std::vector<SphericalPointAndDirection> spherical_points(point_count);
    std::vector<uint32_t> cell_of_point(point_count);
    cell_start_.assign(cell_count_ + 1, 0u);

    for (size_t p = 0; p < point_count; ++p) {
      spherical_points[p] =
          SphericalPointAndDirection(cloud.at(p).getVector3fMap());
      const uint32_t cell_index = static_cast<uint32_t>(CellIndex(
          spherical_points[p].azimuth, spherical_points[p].inclination));
      cell_of_point[p] = cell_index;
      ++cell_start_[cell_index + 1];
    }
    for (size_t cell = 0; cell < cell_count_; ++cell) {
      cell_start_[cell + 1] += cell_start_[cell];
    }

    directions_.assign(3 * point_count, 0.f);
    points_.assign(4 * point_count, 0.f);

    // Counting sort into the flat arrays. Points keep their relative order
    // within a cell, which is the order upstream's push_back() produced.
    std::vector<uint32_t> cursor(cell_start_.begin(), cell_start_.end() - 1);
    for (size_t p = 0; p < point_count; ++p) {
      const size_t target = cursor[cell_of_point[p]]++;
      const SphericalPointAndDirection& sp = spherical_points[p];
      directions_[3 * target + 0] = sp.direction.x();
      directions_[3 * target + 1] = sp.direction.y();
      directions_[3 * target + 2] = sp.direction.z();
      points_[4 * target + 0] = sp.point.x();
      points_[4 * target + 1] = sp.point.y();
      points_[4 * target + 2] = sp.point.z();
      points_[4 * target + 3] = sp.radius;
    }
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

  inline int cell_count_azimuth() const { return cell_count_azimuth_; }

  // First / one-past-last point index of the run of cells
  // [first_cell, last_cell] (which must lie within one inclination row).
  inline uint32_t run_begin(int cell) const { return cell_start_[cell]; }
  inline uint32_t run_end(int cell) const { return cell_start_[cell + 1]; }

  inline const float* directions() const { return directions_.data(); }
  inline const float* points() const { return points_.data(); }

 private:
  int cell_count_azimuth_;
  float cell_extent_azimuth_;
  int cell_count_inclination_;
  float cell_extent_inclination_;
  size_t cell_count_;

  // Indexed by cell index; size cell_count_ + 1.
  std::vector<uint32_t> cell_start_;
  std::vector<float> directions_;
  std::vector<float> points_;
};

// Everything the SIMD filter needs about the reconstruction point under test.
struct BeamFilter {
  // The reconstruction point, in the coordinate frame of the scan.
  float rx, ry, rz;
  // A candidate may be rejected outright if its signed distance along the scan
  // ray is below this value (see ComputeBeamFilter()).
  float min_signed_distance;
  // ... or if its squared distance to the scan ray is above this value.
  float max_ray_distance_squared;
};

// Derives the conservative rejection thresholds for one reconstruction point.
//
// WHY THIS IS EXACT. The filter computes an approximation s~ of the signed
// distance along the ray and an approximation d2~ of the squared distance to
// the ray. It may use a different association and FMA contraction than the
// scalar code, so s~ and d2~ are not bit-identical to the scalar s and d2 --
// they are only close. The filter is therefore used in one direction only: it
// rejects a candidate when the bounds below prove that the *scalar* code would
// also have rejected it. Anything else is handed to EvaluateCandidateExact(),
// which recomputes s and d2 with the unmodified upstream expressions and makes
// the actual decision. Both thresholds use comparisons that are false for NaN,
// so a NaN anywhere sends the candidate down the exact path as well.
//
// The bounds, with u = 2^-24 the float unit roundoff and A = |rx|+|ry|+|rz|:
//   * |s~ - s| <= 6.2 u A  (two 3-term dot products of a unit direction with
//     the point, each with error <= 3.1 u sum|d_i r_i| <= 3.1 u A). The
//     threshold uses 1e-6 A > 3.7e-7 A, a factor 2.7 of slack.
//   * the vector r - s d differs between the two evaluations by at most
//     1e-6 A + 5.5 u A < 1.33e-6 A; the threshold uses 2e-6 A.
//   * d2~ and d2 each carry a relative error <= 3u, absorbed by the 1.0001
//     factor, which is 10x the 1e-5 that the derivation needs.
// Hence d2~ > max_ray_distance_squared implies scalar d2 > beam_radius_squared,
// and s~ < min_signed_distance implies scalar s < 0.
inline BeamFilter ComputeBeamFilter(
    const Eigen::Vector3f& cartesian_reconstruction_point, float beam_radius) {
  BeamFilter filter;
  filter.rx = cartesian_reconstruction_point.x();
  filter.ry = cartesian_reconstruction_point.y();
  filter.rz = cartesian_reconstruction_point.z();

  const float l1_norm = std::fabs(filter.rx) + std::fabs(filter.ry) +
                        std::fabs(filter.rz);
  const float signed_distance_error = 1e-6f * l1_norm + 1e-25f;
  const float ray_offset_error = 2e-6f * l1_norm + 1e-25f;

  filter.min_signed_distance = -signed_distance_error;
  const float max_ray_distance = (beam_radius + ray_offset_error) * 1.0001f;
  filter.max_ray_distance_squared = max_ray_distance * max_ray_distance;
  return filter;
}

// ---------------------------------------------------------------------------
// The blocked filter kernel: three implementations, identical structure.
//
// Input:  ETH3D_CLASSIFY_LANES consecutive scan point directions, interleaved
//         as x,y,z (this is what makes the NEON de-interleaving load free).
// Output: a bit mask whose bit k is set iff lane k could not be rejected and
//         must be evaluated exactly.
// ---------------------------------------------------------------------------
#if defined(__AVX2__) && defined(__FMA__)

// De-interleaves 4 xyz triples held in three 128-bit registers.
//   v0 = [x0 y0 z0 x1], v1 = [y1 z1 x2 y2], v2 = [z2 x3 y3 z3]
inline void Deinterleave3x4(__m128 v0, __m128 v1, __m128 v2, __m128* x,
                            __m128* y, __m128* z) {
  // [x0 x1 x2 x2] then overwrite lane 3 with v2[1] = x3.
  __m128 tx = _mm_shuffle_ps(v0, v1, _MM_SHUFFLE(2, 2, 3, 0));
  *x = _mm_insert_ps(tx, v2, _MM_MK_INSERTPS_NDX(1, 3, 0));
  // [y0 y0 y1 y2] -> [y0 y1 y2 y2] then overwrite lane 3 with v2[2] = y3.
  __m128 ty = _mm_shuffle_ps(v0, v1, _MM_SHUFFLE(3, 0, 1, 1));
  ty = _mm_shuffle_ps(ty, ty, _MM_SHUFFLE(3, 3, 2, 0));
  *y = _mm_insert_ps(ty, v2, _MM_MK_INSERTPS_NDX(2, 3, 0));
  // [z0 z0 z1 z1] -> [z0 z1 z2 z3].
  __m128 tz = _mm_shuffle_ps(v0, v1, _MM_SHUFFLE(1, 1, 2, 2));
  *z = _mm_shuffle_ps(tz, v2, _MM_SHUFFLE(3, 0, 2, 0));
}

inline unsigned FilterCandidateBlock(const float* interleaved_directions,
                                     const BeamFilter& filter) {
  __m128 lo_x, lo_y, lo_z, hi_x, hi_y, hi_z;
  Deinterleave3x4(_mm_loadu_ps(interleaved_directions + 0),
                  _mm_loadu_ps(interleaved_directions + 4),
                  _mm_loadu_ps(interleaved_directions + 8), &lo_x, &lo_y,
                  &lo_z);
  Deinterleave3x4(_mm_loadu_ps(interleaved_directions + 12),
                  _mm_loadu_ps(interleaved_directions + 16),
                  _mm_loadu_ps(interleaved_directions + 20), &hi_x, &hi_y,
                  &hi_z);
  const __m256 dx = _mm256_set_m128(hi_x, lo_x);
  const __m256 dy = _mm256_set_m128(hi_y, lo_y);
  const __m256 dz = _mm256_set_m128(hi_z, lo_z);

  const __m256 rx = _mm256_set1_ps(filter.rx);
  const __m256 ry = _mm256_set1_ps(filter.ry);
  const __m256 rz = _mm256_set1_ps(filter.rz);

  // s = dot(direction, reconstruction point).
  const __m256 s = _mm256_fmadd_ps(
      dz, rz, _mm256_fmadd_ps(dy, ry, _mm256_mul_ps(dx, rx)));
  // offset = reconstruction point - s * direction.
  const __m256 ex = _mm256_fnmadd_ps(s, dx, rx);
  const __m256 ey = _mm256_fnmadd_ps(s, dy, ry);
  const __m256 ez = _mm256_fnmadd_ps(s, dz, rz);
  const __m256 d2 = _mm256_fmadd_ps(
      ez, ez, _mm256_fmadd_ps(ey, ey, _mm256_mul_ps(ex, ex)));

  const __m256 rejected = _mm256_or_ps(
      _mm256_cmp_ps(s, _mm256_set1_ps(filter.min_signed_distance),
                    _CMP_LT_OQ),
      _mm256_cmp_ps(d2, _mm256_set1_ps(filter.max_ray_distance_squared),
                    _CMP_GT_OQ));
  return (~static_cast<unsigned>(_mm256_movemask_ps(rejected))) & 0xffu;
}

#elif defined(__ARM_NEON) && defined(__aarch64__)

inline unsigned FilterCandidateBlock(const float* interleaved_directions,
                                     const BeamFilter& filter) {
  // vld3q_f32 de-interleaves 4 xyz triples in the load itself.
  const float32x4x3_t d = vld3q_f32(interleaved_directions);
  const float32x4_t dx = d.val[0];
  const float32x4_t dy = d.val[1];
  const float32x4_t dz = d.val[2];

  const float32x4_t rx = vdupq_n_f32(filter.rx);
  const float32x4_t ry = vdupq_n_f32(filter.ry);
  const float32x4_t rz = vdupq_n_f32(filter.rz);

  // s = dot(direction, reconstruction point).
  const float32x4_t s =
      vfmaq_f32(vfmaq_f32(vmulq_f32(dx, rx), dy, ry), dz, rz);
  // offset = reconstruction point - s * direction.
  const float32x4_t ex = vfmsq_f32(rx, s, dx);
  const float32x4_t ey = vfmsq_f32(ry, s, dy);
  const float32x4_t ez = vfmsq_f32(rz, s, dz);
  const float32x4_t d2 =
      vfmaq_f32(vfmaq_f32(vmulq_f32(ex, ex), ey, ey), ez, ez);

  const uint32x4_t rejected = vorrq_u32(
      vcltq_f32(s, vdupq_n_f32(filter.min_signed_distance)),
      vcgtq_f32(d2, vdupq_n_f32(filter.max_ray_distance_squared)));
  static const uint32_t kLaneBits[4] = {1u, 2u, 4u, 8u};
  return vaddvq_u32(
      vandq_u32(vmvnq_u32(rejected), vld1q_u32(kLaneBits)));
}

#else

inline unsigned FilterCandidateBlock(const float* interleaved_directions,
                                     const BeamFilter& filter) {
  float dx[ETH3D_CLASSIFY_LANES], dy[ETH3D_CLASSIFY_LANES],
      dz[ETH3D_CLASSIFY_LANES], s[ETH3D_CLASSIFY_LANES],
      d2[ETH3D_CLASSIFY_LANES];
  for (int lane = 0; lane < ETH3D_CLASSIFY_LANES; ++lane) {
    dx[lane] = interleaved_directions[3 * lane + 0];
    dy[lane] = interleaved_directions[3 * lane + 1];
    dz[lane] = interleaved_directions[3 * lane + 2];
  }
  for (int lane = 0; lane < ETH3D_CLASSIFY_LANES; ++lane) {
    // s = dot(direction, reconstruction point).
    s[lane] = dx[lane] * filter.rx + dy[lane] * filter.ry +
              dz[lane] * filter.rz;
  }
  for (int lane = 0; lane < ETH3D_CLASSIFY_LANES; ++lane) {
    // offset = reconstruction point - s * direction.
    const float ex = filter.rx - s[lane] * dx[lane];
    const float ey = filter.ry - s[lane] * dy[lane];
    const float ez = filter.rz - s[lane] * dz[lane];
    d2[lane] = ex * ex + ey * ey + ez * ez;
  }
  unsigned mask = 0;
  for (int lane = 0; lane < ETH3D_CLASSIFY_LANES; ++lane) {
    const bool rejected = (s[lane] < filter.min_signed_distance) ||
                          (d2[lane] > filter.max_ray_distance_squared);
    mask |= static_cast<unsigned>(!rejected) << lane;
  }
  return mask;
}

#endif

// Tests one candidate scan point against the reconstruction point, using the
// unmodified upstream arithmetic. Returns true if the caller must exit early
// because the point is accurate for the smallest tolerance.
inline bool EvaluateCandidateExact(
    const float* direction_xyz, const float* point_xyz_radius,
    const Eigen::Vector3f& cartesian_reconstruction_point,
    // Must be sorted in increasing order.
    const std::vector<float>& accuracy_tolerances_squared,
    float beam_radius_squared, int* first_accurate_tolerance_index,
    bool* inaccurate_classifications_exist) {
  const Eigen::Vector3f scan_point_direction(direction_xyz[0],
                                             direction_xyz[1],
                                             direction_xyz[2]);

  // Is the reconstruction point within the beam volume? (Checked by testing
  // whether the scan ray is closer than beam_radius to the reconstruction
  // point).
  float signed_distance_along_ray =
      scan_point_direction.dot(cartesian_reconstruction_point);
  if (signed_distance_along_ray < 0) {
    // Treat points on the opposite side of the scan ray as unobserved.
    return false;
  }
  Eigen::Vector3f closest_point_on_scan_ray =
      signed_distance_along_ray * scan_point_direction;
  float scan_ray_distance_squared =
      (cartesian_reconstruction_point - closest_point_on_scan_ray)
          .squaredNorm();
  if (scan_ray_distance_squared <= beam_radius_squared) {
    const Eigen::Vector3f scan_point_position(
        point_xyz_radius[0], point_xyz_radius[1], point_xyz_radius[2]);

    // Is the reconstruction point within the region for accurate
    // classification (i.e., closer to the scan point than the evaluation
    // threshold)? In this case, early exit with accurate classification.
    float distance_from_scan_point_squared =
        (scan_point_position - cartesian_reconstruction_point).squaredNorm();
    for (size_t tolerance_index = 0;
         tolerance_index < accuracy_tolerances_squared.size() &&
         static_cast<int>(tolerance_index) < *first_accurate_tolerance_index;
         ++tolerance_index) {
      if (distance_from_scan_point_squared <=
          accuracy_tolerances_squared[tolerance_index]) {
        *first_accurate_tolerance_index = tolerance_index;
        if (tolerance_index == 0) {
          // Early exit.
          return true;
        }
        break;
      }
    }

    // Is the reconstruction point in front of the scan point? In this case,
    // remember that inaccurate classifications may exist.
    if (signed_distance_along_ray < point_xyz_radius[3]) {
      *inaccurate_classifications_exist = true;
    }
  }
  return false;
}

// Classifies the reconstruction point against the contiguous run of scan
// points [begin, end). Whole blocks go through the SIMD rejection filter;
// the remainder is evaluated exactly right away, which is cheaper than
// filtering it first. Returns true if the caller must exit early.
inline bool ClassifyAgainstRun(
    const ScanPointGrid& point_grid, uint32_t begin, uint32_t end,
    const Eigen::Vector3f& cartesian_reconstruction_point,
    const BeamFilter& filter,
    // Must be sorted in increasing order.
    const std::vector<float>& accuracy_tolerances_squared,
    float beam_radius_squared, int* first_accurate_tolerance_index,
    bool* inaccurate_classifications_exist) {
  const float* directions = point_grid.directions();
  const float* points = point_grid.points();

  uint32_t index = begin;
  for (; index + ETH3D_CLASSIFY_LANES <= end;
       index += ETH3D_CLASSIFY_LANES) {
    unsigned surviving_lanes =
        FilterCandidateBlock(directions + 3 * static_cast<size_t>(index),
                             filter);
    while (surviving_lanes != 0) {
      const unsigned lane = __builtin_ctz(surviving_lanes);
      surviving_lanes &= surviving_lanes - 1;
      const size_t candidate = static_cast<size_t>(index) + lane;
      if (EvaluateCandidateExact(directions + 3 * candidate,
                                 points + 4 * candidate,
                                 cartesian_reconstruction_point,
                                 accuracy_tolerances_squared,
                                 beam_radius_squared,
                                 first_accurate_tolerance_index,
                                 inaccurate_classifications_exist)) {
        return true;
      }
    }
  }
  for (; index < end; ++index) {
    const size_t candidate = index;
    if (EvaluateCandidateExact(directions + 3 * candidate,
                               points + 4 * candidate,
                               cartesian_reconstruction_point,
                               accuracy_tolerances_squared,
                               beam_radius_squared,
                               first_accurate_tolerance_index,
                               inaccurate_classifications_exist)) {
      return true;
    }
  }
  return false;
}

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
                          const ScanPointGrid& point_grid,
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
  // direction. This visits exactly the cells that upstream's
  // SphericalPointGridIterator visited (a cell that the iterator visited twice,
  // which only happens when the azimuth range covers the full circle, is
  // visited once here -- the two results this function computes are a minimum
  // and a logical or, so a repeated cell cannot change them).
  int min_cell_index_azimuth, max_cell_index_azimuth;
  int min_cell_index_inclination, max_cell_index_inclination;
  point_grid.CellCoordinatesWithoutWrap(
      spherical_reconstruction_point.azimuth - relevancy_angle_horizontal,
      spherical_reconstruction_point.inclination - relevancy_angle_vertical,
      &min_cell_index_azimuth, &min_cell_index_inclination);
  point_grid.CellCoordinatesWithoutWrap(
      spherical_reconstruction_point.azimuth + relevancy_angle_horizontal,
      spherical_reconstruction_point.inclination + relevancy_angle_vertical,
      &max_cell_index_azimuth, &max_cell_index_inclination);

  const BeamFilter filter =
      ComputeBeamFilter(cartesian_reconstruction_point, beam_radius);

  const int cell_count_azimuth = point_grid.cell_count_azimuth();
  const int azimuth_cell_count =
      max_cell_index_azimuth - min_cell_index_azimuth + 1;

  for (int cell_index_inclination = min_cell_index_inclination;
       cell_index_inclination <= max_cell_index_inclination;
       ++cell_index_inclination) {
    const int row_first_cell = cell_count_azimuth * cell_index_inclination;

    // The scan points of consecutive azimuth cells of one inclination row are
    // consecutive in the grid's flat arrays, so the whole row range is a single
    // run of points -- two runs if it wraps around the +-pi discontinuity.
    int run_first_azimuth[2];
    int run_last_azimuth[2];
    int run_count;
    if (azimuth_cell_count >= cell_count_azimuth) {
      // The range covers the full circle.
      run_first_azimuth[0] = 0;
      run_last_azimuth[0] = cell_count_azimuth - 1;
      run_count = 1;
    } else {
      const int first = mod(min_cell_index_azimuth, cell_count_azimuth);
      const int last = first + azimuth_cell_count - 1;
      if (last < cell_count_azimuth) {
        run_first_azimuth[0] = first;
        run_last_azimuth[0] = last;
        run_count = 1;
      } else {
        run_first_azimuth[0] = first;
        run_last_azimuth[0] = cell_count_azimuth - 1;
        run_first_azimuth[1] = 0;
        run_last_azimuth[1] = last - cell_count_azimuth;
        run_count = 2;
      }
    }

    for (int run = 0; run < run_count; ++run) {
      const uint32_t begin =
          point_grid.run_begin(row_first_cell + run_first_azimuth[run]);
      const uint32_t end =
          point_grid.run_end(row_first_cell + run_last_azimuth[run]);
      if (begin == end) {
        continue;
      }
      if (ClassifyAgainstRun(point_grid, begin, end,
                             cartesian_reconstruction_point, filter,
                             accuracy_tolerances_squared, beam_radius_squared,
                             first_accurate_tolerance_index,
                             inaccurate_classifications_exist)) {
        // Early exit: the point is accurate for the smallest tolerance.
        return;
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
  // cells defined on the spherical coordinates.
  std::vector<std::unique_ptr<ScanPointGrid>> point_grids(scan_count);
  for (size_t scan_index = 0; scan_index < scan_count; ++scan_index) {
    point_grids[scan_index].reset(
        new ScanPointGrid(kCellCountAzimuth, kCellCountInclination));
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
  // Indexed by: [map_index][CalcCellCoordinates(...)][tolerance_index].
  std::unordered_map<std::tuple<int, int, int>, std::vector<AccuracyCell>>
      cell_maps[kGridCount];

  // Loop over the reconstruction points.
  for (size_t point_index = 0, size = reconstruction.size(); point_index < size;
       ++point_index) {
    const pcl::PointXYZ& point = reconstruction.at(point_index);

    // Find the voxels for this reconstruction point.
    std::vector<AccuracyCell>* cell_vectors[kGridCount];
    for (int grid_index = 0; grid_index < kGridCount; ++grid_index) {
      std::vector<AccuracyCell>* cell_vector =
          &cell_maps[grid_index][CalcCellCoordinates(
              point, voxel_size_inv, kGridShifts[grid_index][0],
              kGridShifts[grid_index][1], kGridShifts[grid_index][2])];
      if (cell_vector->empty()) {
        cell_vector->resize(tolerances_count);
      }
      cell_vectors[grid_index] = cell_vector;
    }

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
      const ScanPointGrid* point_grid = point_grids[scan_index].get();
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

    // Aggregate accurate count.
    for (int tolerance_index = aggregate_first_accurate_tolerance_index;
         tolerance_index < static_cast<int>(tolerances_count);
         ++tolerance_index) {
      for (int grid_index = 0; grid_index < kGridCount; ++grid_index) {
        ++cell_vectors[grid_index]->at(tolerance_index).accurate_count;
      }
      if (output_point_results) {
        point_is_accurate->at(tolerance_index)[point_index] =
            AccuracyResult::kAccurate;
      }
    }
    // Aggregate inaccurate count or unobserved count.
    if (aggregate_inaccurate_classifications_exist) {
      for (int tolerance_index = aggregate_first_accurate_tolerance_index - 1;
           tolerance_index >= 0; --tolerance_index) {
        for (int grid_index = 0; grid_index < kGridCount; ++grid_index) {
          ++cell_vectors[grid_index]->at(tolerance_index).inaccurate_count;
        }
        if (output_point_results) {
          point_is_accurate->at(tolerance_index)[point_index] =
              AccuracyResult::kInaccurate;
        }
      }
    } else {
      if (output_point_results) {
        for (int tolerance_index = aggregate_first_accurate_tolerance_index - 1;
             tolerance_index >= 0; --tolerance_index) {
          point_is_accurate->at(tolerance_index)[point_index] =
              AccuracyResult::kUnobserved;
        }
      }
    }
  }

  // Average results over all cells and fill the results vector.
  std::vector<double> accuracy_sum(tolerances_count, 0.0);
  std::vector<size_t> valid_cell_count(tolerances_count, 0);
  for (int grid_index = 0; grid_index < kGridCount; ++grid_index) {
    for (auto it = cell_maps[grid_index].cbegin(),
              end = cell_maps[grid_index].cend();
         it != end; ++it) {
      const std::vector<AccuracyCell>& cell_vector = it->second;
      for (size_t tolerance_index = 0; tolerance_index < tolerances_count;
           ++tolerance_index) {
        const AccuracyCell& cell = cell_vector[tolerance_index];
        size_t valid_point_count = cell.accurate_count + cell.inaccurate_count;
        if (valid_point_count > 0) {
          accuracy_sum[tolerance_index] +=
              cell.accurate_count / (1.0f * valid_point_count);
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
