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

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "util.h"

// A uniform-voxel nearest-neighbour index over the reconstruction cloud, in CSR
// layout.
//
// It answers exactly the question ComputeCompleteness asks and nothing more:
// "what is the squared distance from this scan point to the closest
// reconstruction point, if that distance is below the largest evaluation
// tolerance?". The caller reads one float and reduces it to a tolerance band
// with a strict '<' against the squared maximum tolerance, so any answer at or
// above that threshold is interchangeable with +infinity. That is what lets
// this index stop expanding at a fixed radius; a kd-tree cannot exploit it and
// has to descend the whole tree for every query.
//
// WHY THIS REPLACES pcl::search::KdTree
// -------------------------------------
// Building the FLANN kd-tree over 2.79M points is a single-threaded 0.54 s,
// which was 39.6% of the whole program's 12-thread wall clock and could not be
// parallelised without reaching inside libpcl. This index is built by a
// counting sort whose three passes are all parallel, and whose result does not
// depend on the thread count at all.
//
// WHY THE ANSWER IS BIT-IDENTICAL TO FLANN'S
// ------------------------------------------
// Three separate claims, and the class is written so that each is checkable:
//
// (1) SEARCH COMPLETENESS. The probe examines cells in Chebyshev shells around
//     the query's own cell and stops only once the shell boundary is provably
//     farther away than the best candidate found so far. A point stored in a
//     cell lies inside that cell's half-open box by construction (the cell
//     index IS the floor of the scaled coordinate), so the distance from the
//     query to a cell's box lower-bounds the distance to every point in it.
//     Every lower bound is compared against a threshold widened by
//     kBoundSlack, so rounding in the bound can only make the probe examine
//     more cells, never fewer.
//
//     That invariant has a domain, and the domain is enforced from BOTH
//     sides, which is the part it is easy to get wrong. From above, the grid
//     is never made finer than the exponent for which a scaled coordinate
//     could leave the range where double arithmetic on it is exact
//     (kMaxGridCoordinate). From below, the cell-size search only ever
//     terminates once the cell count is inside the budget, and the budget is
//     small enough that every axis dimension, every flat cell id and the cell
//     array's own size are inside the integer types that hold them. A search
//     that could end on the coarse side WITHOUT meeting the budget would wrap
//     a uint32 cell id, file points in cells that do not contain them, and
//     break the containment invariant just as thoroughly as an over-fine grid
//     -- so Build refuses to return an index at all rather than return one it
//     cannot make exact, and the caller answers from the kd-tree instead.
//
// (2) THE SEARCH RADIUS CAP. Once the shell boundary is at or beyond the
//     maximum tolerance, every unexamined point has a squared distance at or
//     above the caller's squared threshold, and the caller's test is a strict
//     '<'. Returning +infinity instead of that point's actual distance is
//     therefore invisible: both fail the test.
//
// (3) THE VALUE ITSELF. The squared distance kernel below is the literal
//     statement sequence of flann::L2_Simple::operator() -- a float
//     accumulator, three terms, strictly left to right -- so the compiler
//     generates the same rounding sequence (on arm64, the same `fmadd` chain
//     that a disassembly of the installed libpcl_kdtree shows FLANN itself
//     compiled to). Ties do not matter because the caller never reads the
//     neighbour's index, only its distance.
//
// Claim (3) is the only one that rests on codegen rather than on proof, which
// is why --nn_verify exists: it runs both indices over every scan point and
// compares them bit for bit. Run it once on any new host or scene.
class ReconNnGrid {
 public:
  // Builds the index over the finite points of `cloud`.
  //
  // Non-finite points are dropped, which is exactly what
  // pcl::KdTreeFLANN::convertCloudToArray does (it skips every point for which
  // the point representation is not valid, whatever the cloud's is_dense flag
  // says), so the candidate set is identical. It would be harmless anyway: a
  // squared distance to a non-finite point is NaN, and every comparison FLANN
  // makes against it is a strict '<' that NaN loses.
  //
  // `maximum_tolerance` and `maximum_tolerance_squared` are the caller's
  // largest tolerance and the exact float it compares against; the probe uses
  // them to stop expanding once no further point could pass that test.
  //
  // Returns false if no grid satisfying the index's own domain bounds exists
  // for this cloud, in which case the object is left empty and MUST NOT be
  // queried -- the caller has to fall back to the kd-tree. No cloud of finite
  // float coordinates can trigger that (see kMinGridExponent in the .cc), so
  // the path is a backstop for a broken assumption rather than a case the
  // campaign will meet; it exists because the alternative to refusing is
  // answering with cell ids that have wrapped, which is a wrong number that
  // looks like a result.
  bool Build(const PointCloud& cloud, float maximum_tolerance,
             float maximum_tolerance_squared);

  // Squared distance to the closest point of the cloud, or +infinity if no
  // point is close enough to matter (see claim (2) above).
  inline float NearestSquaredDistance(float qx, float qy, float qz) const;

  // Edge length of a voxel, and the number of occupied/total cells. Only used
  // for the --nn_verify report.
  inline double cell_size() const { return cell_size_; }
  inline uint64_t cell_count() const { return total_cell_count_; }
  inline size_t point_count() const { return point_xyz_.size() / 3; }

 private:
  // Relative widening applied to every geometric lower bound before it is
  // compared against a squared distance. The bounds are computed in double
  // from float inputs, so their own error is ~1e-16 relative; the float
  // distance kernel's error against exact arithmetic is below 2e-7 relative.
  // 1e-5 dominates both by more than an order of magnitude, and it is applied
  // in the direction that makes the probe do MORE work, so it cannot cause a
  // point to be missed. The cost is that the probe occasionally scans one cell
  // it could have skipped.
  static constexpr double kBoundSlack = 1.0 + 1e-5;

  // Scaled grid coordinate of a world coordinate: cell index = floor(this).
  // `origin` is an integer multiple of the cell size and `inv_cell_size` is a
  // power of two, so for any coordinate a scene can plausibly hold this is
  // computed exactly in double, which is what makes "a point in cell i has its
  // coordinate in [origin + i*h, origin + (i+1)*h)" an exact invariant rather
  // than an approximate one.
  static inline double GridCoordinate(double world, double origin,
                                      double inv_cell_size) {
    return (world - origin) * inv_cell_size;
  }

  // Distance from a query at scaled coordinate `u` to the slab of cell `i`
  // along one axis, in world units. Zero when the query is inside the slab.
  inline double AxisGap(double u, int64_t i) const {
    const double a = u - static_cast<double>(i);
    return cell_size_ * std::max(0.0, std::max(-a, a - 1.0));
  }

  // Distance from a query at scaled coordinate `u` to the whole grid along one
  // axis, in world units; the grid occupies [0, dim) in scaled coordinates.
  inline double GridGap(double u, int64_t dim) const {
    return cell_size_ *
           std::max(0.0, std::max(-u, u - static_cast<double>(dim)));
  }

  double origin_[3];
  double cell_size_;      // h, a power of two.
  double inv_cell_size_;  // 1/h, a power of two.
  int64_t dim_[3];
  uint64_t total_cell_count_;

  // Squared distance at or above which the caller cannot tell one answer from
  // another, as a double so the shell bound can be compared against it.
  double cap_squared_;

  // CSR layout: the points of cell c are the entries
  // [cell_start_[c], cell_start_[c + 1]) of point_xyz_, which stores x, y and z
  // interleaved.
  //
  // Interleaved rather than three parallel arrays because the inner loop reads
  // one short run of points per cell (about thirty on the bench scene) from a
  // 34 MB array at an unpredictable offset, so it is bound by how many distinct
  // cache lines and pages it touches rather than by arithmetic throughput.
  // Interleaving makes a cell's points one contiguous run instead of three runs
  // in three different pages. Measured: 0.397 s against 0.441 s for the
  // three-array layout over the bench scene at one thread.
  std::vector<uint32_t> cell_start_;
  std::vector<float> point_xyz_;
};

inline float ReconNnGrid::NearestSquaredDistance(float qx, float qy,
                                                 float qz) const {
  const float kNoNeighbour = std::numeric_limits<float>::infinity();
  if (point_xyz_.empty()) {
    return kNoNeighbour;
  }

  // A non-finite query makes every squared distance NaN, which loses every
  // strict '<' in FLANN's result set, so FLANN reports nothing found. Report
  // nothing found as well. (The caller's test rejects +infinity and NaN alike,
  // so the classification would be the same even if the two disagreed here.)
  if (!std::isfinite(qx) || !std::isfinite(qy) || !std::isfinite(qz)) {
    return kNoNeighbour;
  }

  const double ux = GridCoordinate(qx, origin_[0], inv_cell_size_);
  const double uy = GridCoordinate(qy, origin_[1], inv_cell_size_);
  const double uz = GridCoordinate(qz, origin_[2], inv_cell_size_);

  // Reject a query that is far outside the grid before touching any cell. This
  // is both a large shortcut for scan points that stick out of the
  // reconstruction and the guarantee that the cell indices below stay in a
  // small range, so the shell loop cannot run away on a query a kilometre from
  // the scene. The grid occupies [0, dim) in scaled coordinates on each axis.
  {
    const double bx = GridGap(ux, dim_[0]);
    const double by = GridGap(uy, dim_[1]);
    const double bz = GridGap(uz, dim_[2]);
    const double box_squared = bx * bx + by * by + bz * bz;
    if (box_squared >= cap_squared_ * kBoundSlack) {
      return kNoNeighbour;
    }
  }

  const int64_t cx = static_cast<int64_t>(std::floor(ux));
  const int64_t cy = static_cast<int64_t>(std::floor(uy));
  const int64_t cz = static_cast<int64_t>(std::floor(uz));

  // Fractional position inside the home cell, and the smallest distance (in
  // cells) from the query to the home cell's own boundary. The boundary of the
  // Chebyshev block of radius r is then exactly (r + boundary_fraction) cells
  // away, which is the lower bound that terminates the expansion.
  const double frac_x = ux - static_cast<double>(cx);
  const double frac_y = uy - static_cast<double>(cy);
  const double frac_z = uz - static_cast<double>(cz);
  const double boundary_fraction =
      std::min(std::min(std::min(frac_x, 1.0 - frac_x),
                        std::min(frac_y, 1.0 - frac_y)),
               std::min(frac_z, 1.0 - frac_z));

  float best = kNoNeighbour;
  // Threshold a geometric lower bound has to reach before the cell or the shell
  // it bounds can be skipped. Kept in step with `best` so every pruning test is
  // a single double compare.
  double prune = std::numeric_limits<double>::infinity();

  const uint32_t* const cell_start = cell_start_.data();
  const float* const pxyz = point_xyz_.data();

  // Scans one cell, given the squared lower bound already accumulated over the
  // y and z axes for its row.
  const auto scan_cell = [&](int64_t row_base, int64_t ix, double bound_yz) {
    const double gap_x = AxisGap(ux, ix);
    const double bound = bound_yz + gap_x * gap_x;
    if (bound >= prune) {
      return;
    }
    const uint32_t begin = cell_start[row_base + ix];
    const uint32_t end = cell_start[row_base + ix + 1];
    for (uint32_t i = begin; i < end; ++i) {
      // The literal accumulation order of flann::L2_Simple::operator(): a float
      // accumulator seeded with zero and three squared differences added left
      // to right. Do not reorder it, do not introduce a second accumulator and
      // do not sum the three products separately -- each of those changes the
      // rounding and with it the last bits of the value the caller compares
      // against a tolerance.
      float d = 0.f;
      float t = qx - pxyz[3 * i];
      d += t * t;
      t = qy - pxyz[3 * i + 1];
      d += t * t;
      t = qz - pxyz[3 * i + 2];
      d += t * t;
      if (d < best) {
        best = d;
        prune = static_cast<double>(d) * kBoundSlack;
      }
    }
  };

  for (int64_t r = 0;; ++r) {
    const int64_t z_lo = std::max<int64_t>(cz - r, 0);
    const int64_t z_hi = std::min<int64_t>(cz + r, dim_[2] - 1);
    for (int64_t iz = z_lo; iz <= z_hi; ++iz) {
      const bool z_on_shell = (iz == cz - r) || (iz == cz + r);
      const double gap_z = AxisGap(uz, iz);
      const double bound_z = gap_z * gap_z;
      if (bound_z >= prune) {
        continue;
      }

      const int64_t y_lo = std::max<int64_t>(cy - r, 0);
      const int64_t y_hi = std::min<int64_t>(cy + r, dim_[1] - 1);
      for (int64_t iy = y_lo; iy <= y_hi; ++iy) {
        const bool y_on_shell = (iy == cy - r) || (iy == cy + r);
        const double gap_y = AxisGap(uy, iy);
        const double bound_yz = bound_z + gap_y * gap_y;
        if (bound_yz >= prune) {
          continue;
        }

        // The cells of one row are contiguous in the flat cell array.
        const int64_t row_base = (iz * dim_[1] + iy) * dim_[0];

        if (z_on_shell || y_on_shell) {
          // The row is already on the shell, so its whole x extent belongs to
          // the shell.
          const int64_t x_lo = std::max<int64_t>(cx - r, 0);
          const int64_t x_hi = std::min<int64_t>(cx + r, dim_[0] - 1);
          for (int64_t ix = x_lo; ix <= x_hi; ++ix) {
            scan_cell(row_base, ix, bound_yz);
          }
        } else {
          // An interior row touches the shell only at its two x faces (one cell
          // when r is zero, but then the row is on the shell anyway).
          if (cx - r >= 0 && cx - r <= dim_[0] - 1) {
            scan_cell(row_base, cx - r, bound_yz);
          }
          if (cx + r >= 0 && cx + r <= dim_[0] - 1) {
            scan_cell(row_base, cx + r, bound_yz);
          }
        }
      }
    }

    // Distance from the query to the boundary of the Chebyshev block of radius
    // r, squared. Every point not examined yet lies outside that block, so this
    // lower-bounds the distance to all of them.
    const double shell_bound =
        cell_size_ * (static_cast<double>(r) + boundary_fraction);
    const double shell_bound_squared = shell_bound * shell_bound;

    // Nothing outside the block can beat what we already have.
    if (shell_bound_squared >= prune) {
      break;
    }
    // Nothing outside the block can pass the caller's tolerance test either, so
    // whatever is out there is interchangeable with what we hold.
    if (shell_bound_squared >= cap_squared_ * kBoundSlack) {
      break;
    }
    // Every cell of the grid has been covered, so `best` is already the exact
    // global minimum. Without this the loop would keep expanding into empty
    // space until the cap fired.
    if (cx - r <= 0 && cx + r >= dim_[0] - 1 && cy - r <= 0 &&
        cy + r >= dim_[1] - 1 && cz - r <= 0 && cz + r >= dim_[2] - 1) {
      break;
    }
  }

  return best;
}
