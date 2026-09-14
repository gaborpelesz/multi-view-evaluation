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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "completeness.h"

#include <pcl/common/transforms.h>
#include <pcl/io/ply_io.h>
#include <pcl/search/kdtree.h>

#ifdef _OPENMP
#include <omp.h>
#endif

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

// ---------------------------------------------------------------------------
// Nearest neighbour index over the reconstruction.
// ---------------------------------------------------------------------------

namespace {

// Upper bound on the number of FLANN sub-indices. Every query scans every
// partition's bounding box, so past a point the scan costs more than the
// shrinking sub-trees save; 16 is comfortably above the core count of the
// machines this is measured on.
constexpr int kMaxPartitions = 16;

// Smallest partition worth creating. A sub-tree over a few thousand points is
// built in microseconds, so splitting further only adds per-query box tests.
constexpr size_t kMinPartitionPoints = 16384;

// Subsets below this size are split inline rather than in their own OpenMP
// task: the task overhead would dominate a median selection this short.
constexpr size_t kSplitTaskThreshold = 65536;

// The multiplier that turns a computed box distance into a bound that is
// guaranteed not to exceed the float squared distance FLANN reports for any
// point inside the box. 2^-18 is roughly a factor of sixteen more slack than
// the worst case of the two error sources it has to cover (see the pruning
// argument on ReconstructionNearestIndex), and costs nothing: it can only make
// the search visit a partition it would otherwise have skipped, and only for
// query points sitting within a relative 4e-6 of a box face.
constexpr double kBoundShrink = 1.0 - 1.0 / 262144.0;

// A partition's axis-aligned bounding box, computed from the points that
// actually landed in it rather than from the splitting planes. A tight box
// prunes more queries, and "every point of the partition is inside this box" is
// the only property the pruning argument needs.
struct PartitionBox {
  float lo[3];
  float hi[3];
};

// One reconstruction point packed next to its cloud index, used only while
// partitioning. std::nth_element moves these 16-byte records around; selecting
// over a permutation of cloud indices instead would turn every comparison into
// a random read somewhere in the 45 MB point cloud.
struct PartitionPoint {
  float c[3];
  pcl::index_t index;
};

// Distance from `v` to the interval [lo, hi], as a double. Both bounds are
// floats, so the promotion and the subtraction are exact.
inline double AxisGap(double v, float lo, float hi) {
  if (v < lo) {
    return static_cast<double>(lo) - v;
  }
  if (v > hi) {
    return v - static_cast<double>(hi);
  }
  return 0.0;
}

// A strict lower bound on the squared distance from `point` to any point of
// `box`. A non-finite coordinate makes both comparisons in AxisGap false and so
// yields a bound of zero, which is conservative: it prunes nothing.
inline double BoxDistanceBound(const pcl::PointXYZ& point,
                               const PartitionBox& box) {
  const double gap_x = AxisGap(point.x, box.lo[0], box.hi[0]);
  const double gap_y = AxisGap(point.y, box.lo[1], box.hi[1]);
  const double gap_z = AxisGap(point.z, box.lo[2], box.hi[2]);
  return (gap_x * gap_x + gap_y * gap_y + gap_z * gap_z) * kBoundShrink;
}

// Splits points[begin, end) into `parts` groups of as equal a point count as
// integer division allows and writes their ranges to ranges[slot, slot + parts).
// Each split cuts the longest axis of the subset's own bounding box at the
// point count median, which keeps the groups compact -- compact groups are what
// make the query-side bound large enough to prune with.
//
// Requires end - begin >= parts, which the caller guarantees via
// kMinPartitionPoints; that is what makes both halves non-empty.
//
// The permutation std::nth_element leaves behind is a function of the subset
// alone, and the two halves are disjoint ranges written to disjoint slots, so
// the partitioning does not depend on how OpenMP schedules the tasks.
void SplitPartitions(PartitionPoint* points, size_t begin, size_t end,
                     int parts, int slot, std::pair<size_t, size_t>* ranges) {
  if (parts <= 1) {
    ranges[slot] = std::make_pair(begin, end);
    return;
  }

  float lo[3] = {points[begin].c[0], points[begin].c[1], points[begin].c[2]};
  float hi[3] = {lo[0], lo[1], lo[2]};
  for (size_t i = begin + 1; i < end; ++i) {
    for (int axis = 0; axis < 3; ++axis) {
      const float value = points[i].c[axis];
      if (value < lo[axis]) {
        lo[axis] = value;
      } else if (value > hi[axis]) {
        hi[axis] = value;
      }
    }
  }
  int split_axis = 0;
  if (hi[1] - lo[1] > hi[split_axis] - lo[split_axis]) {
    split_axis = 1;
  }
  if (hi[2] - lo[2] > hi[split_axis] - lo[split_axis]) {
    split_axis = 2;
  }

  const int left_parts = parts / 2;
  const size_t mid = begin + (end - begin) * static_cast<size_t>(left_parts) /
                                 static_cast<size_t>(parts);
  std::nth_element(points + begin, points + mid, points + end,
                   [split_axis](const PartitionPoint& a,
                                const PartitionPoint& b) {
                     return a.c[split_axis] < b.c[split_axis];
                   });

#pragma omp task if (end - mid > kSplitTaskThreshold)
  SplitPartitions(points, mid, end, parts - left_parts, slot + left_parts,
                  ranges);
  SplitPartitions(points, begin, mid, left_parts, slot, ranges);
#pragma omp taskwait
}

// The nearest neighbour index ComputeCompleteness queries, in either of two
// shapes selected by --nn_index.
//
// kFlann is the original: one pcl::search::KdTree over the whole
// reconstruction. FLANN's kd-tree build contains no OpenMP at all, so that one
// call is single-threaded no matter how many cores are available -- 0.54 s on
// the benchmark scene, more than a third of the entire 12-thread run.
//
// kPartitioned splits the reconstruction into P spatially disjoint groups and
// gives each its own pcl::search::KdTree, all built inside one parallel loop.
// A query bounds the distance from the query point to each group's bounding
// box, searches the group with the smallest bound, and searches a further group
// only while its bound is still below the best squared distance found so far.
//
// WHY BOTH SHAPES RETURN THE SAME FLOAT, BIT FOR BIT
//
//  * No floating point arithmetic is added to the value path. Every candidate
//    distance is still computed by the same precompiled flann::L2_Simple<float>
//    inside libpcl_kdtree, over coordinates copied verbatim by
//    KdTreeFLANN::convertCloudToArray, from the same KDTreeSingleIndexParams(15)
//    and the same SearchParams(-1, 0.0f). The only operation this class adds is
//    a minimum over floats, which is exact and order independent.
//  * The indexed convertCloudToArray overload applies the same isValid filter
//    as the unindexed one, and the partitions are built from the points that
//    pass that filter, so the candidate set is exactly the original one.
//  * Every point of partition j lies inside box j, so no point there is closer
//    to the query than the box bound. A partition is skipped only when that
//    bound -- deliberately under-estimated, see kBoundShrink -- already exceeds
//    the best distance found, which cannot discard a strictly smaller value;
//    and the partition holding the true nearest point can never be skipped,
//    because its bound is at most that point's distance, which is at most the
//    running best.
//  * Ties may resolve to a different point index than one tree would have
//    picked. Nothing reads the index: the caller uses only the distance.
//
// WHAT THAT ARGUMENT DOES NOT COVER, STATED PLAINLY
//
// FLANN runs a branch and bound of its own, on a float this code cannot reach.
// searchLevel (kdtree_single_index.h:636) carries the squared distance to the
// box of the node it is visiting as
//     mindistsq = mindistsq + cut_dist - dst
// and prunes the far child at :638 when that exceeds the best distance found so
// far. It is a float running sum with a cancelling subtraction, so it is a
// lower bound only up to rounding, and a value that rounds high on a close
// enough tie would prune a subtree holding a point at that same distance and
// make the tree report a distance that is too large. That is FLANN's own
// behaviour, present in the original single-index path too -- but partitioning
// changes the regime the sum runs in, and the honest thing is to say so rather
// than to call it one more draw from the same distribution.
// computeInitialDistances (:571) returns exactly zero, with a zeroed per-axis
// vector, whenever the query lies inside the tree's root box; against one tree
// over the whole reconstruction essentially every scan point does, so the first
// update on each axis has dst == 0 and is exact. Against P sub-trees the query
// lies outside the root box of P-1 of them by construction, so the per-axis
// vector starts nonzero and the cancellation begins at the first level down.
// kBoundShrink protects the box test this class performs; it cannot protect a
// bound computed inside the library.
//
// What bounds that risk, none of it a proof:
//  * The direction is one-sided. A bound that rounds high can only make a
//    sub-tree report a distance that is too large, never too small, and the
//    answer is the minimum over the sub-trees searched. A divergence therefore
//    needs the mis-prune to happen in the one partition that holds the true
//    nearest neighbour -- and that is almost always the partition whose box the
//    query lies inside, i.e. the same exact-start regime as the single tree.
//  * It has been looked for and not found. --nn_index both reports zero
//    disagreements over the 3.17M queries of the benchmark scene, over the
//    tiny scene forced to sixteen partitions, and over hollow shell scenes
//    built so that the average query searches 5.5 of 12 sub-trees -- scenes,
//    that is, that put most of their sub-searches in the nonzero-start regime
//    described above. An independent review found none either, over ~2.9M
//    further queries on synthetic scenes carrying non-finite coordinates,
//    coincident points, collinear and planar partitions, denormal and overflow
//    scale coordinates, an exact-tolerance lattice, and coordinates 1e5 m from
//    the origin.
//  * --nn_index both runs both shapes over the same scan and reports any
//    disagreement, so a new host or an unfamiliar scene is one run away from an
//    answer; --nn_index flann is the way out if one ever fires.
// Closing it by proof would mean vendoring FLANN, which would recompile
// L2_Simple with this project's flags and trade a bound that has never been
// observed to move for an FMA contraction risk that certainly would.
class ReconstructionNearestIndex {
 public:
  // `requested_partitions` is an upper bound, not a promise: the index falls
  // back to a single tree over the whole cloud when the reconstruction is too
  // small to be worth splitting, which is also what keeps the single threaded
  // run identical in shape to the original.
  void Build(const PointCloudPtr& cloud, int requested_partitions);

  // Squared distance to the nearest reconstruction point, or +infinity if
  // FLANN reported nothing (which it does for a non-finite query point).
  // The two scratch vectors are the caller's, one set per thread, and must have
  // size 1; they exist only so the hot loop does not reallocate them.
  inline float NearestSquaredDistance(
      const pcl::PointXYZ& point, pcl::Indices* knn_indices,
      std::vector<float>* knn_squared_dists) const {
    if (boxes_.empty()) {
      return QueryTree(0, point, knn_indices, knn_squared_dists);
    }

    const int partition_count = static_cast<int>(trees_.size());
    double bounds[kMaxPartitions];
    int nearest_partition = 0;
    for (int i = 0; i < partition_count; ++i) {
      bounds[i] = BoxDistanceBound(point, boxes_[i]);
      if (bounds[i] < bounds[nearest_partition]) {
        nearest_partition = i;
      }
    }

    // Searching the closest box first is what makes the loop below skip the
    // rest: after it, `best` is already a real neighbour distance, typically
    // millimetres against metre-sized boxes.
    float best = QueryTree(nearest_partition, point, knn_indices,
                           knn_squared_dists);
    for (int i = 0; i < partition_count; ++i) {
      if (i == nearest_partition || !(bounds[i] < static_cast<double>(best))) {
        continue;
      }
      const float candidate = QueryTree(i, point, knn_indices,
                                        knn_squared_dists);
      if (candidate < best) {
        best = candidate;
      }
    }
    return best;
  }

 private:
  inline float QueryTree(int partition, const pcl::PointXYZ& point,
                         pcl::Indices* knn_indices,
                         std::vector<float>* knn_squared_dists) const {
    // pcl::KdTreeFLANN::nearestKSearch returns min(k, cloud size)
    // unconditionally rather than the number of neighbours actually written, so
    // its return value cannot be used to detect "nothing found" (which FLANN
    // does produce for a non-finite query point). Pre-seeding the slot with
    // infinity makes that case fail the tolerance test, which is exactly what
    // radiusSearch returning 0 did.
    (*knn_squared_dists)[0] = std::numeric_limits<float>::infinity();
    trees_[partition]->nearestKSearch(point, 1, *knn_indices,
                                      *knn_squared_dists);
    return (*knn_squared_dists)[0];
  }

  // One tree when boxes_ is empty, otherwise one tree per box.
  std::vector<std::shared_ptr<pcl::search::KdTree<pcl::PointXYZ>>> trees_;
  std::vector<PartitionBox> boxes_;
  // Kept alive because pcl::search::KdTree holds the index list it was built
  // from by reference, not by value.
  std::vector<pcl::IndicesPtr> partition_indices_;
};

void ReconstructionNearestIndex::Build(const PointCloudPtr& cloud,
                                       int requested_partitions) {
  if (requested_partitions > 1 &&
      cloud->size() >= 2 * kMinPartitionPoints) {
    // Collect the points FLANN would actually index. KdTreeFLANN drops
    // non-finite points -- convertCloudToArray applies isValid on both its
    // indexed and its unindexed overload -- so dropping them here changes
    // nothing about the candidate set, while guaranteeing that every partition
    // below has a finite bounding box and at least one point in it.
    std::vector<PartitionPoint> points;
    points.reserve(cloud->size());
    for (size_t i = 0; i < cloud->size(); ++i) {
      const pcl::PointXYZ& point = cloud->at(i);
      if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
          !std::isfinite(point.z)) {
        continue;
      }
      PartitionPoint packed;
      packed.c[0] = point.x;
      packed.c[1] = point.y;
      packed.c[2] = point.z;
      packed.index = static_cast<pcl::index_t>(i);
      points.push_back(packed);
    }

    int partition_count = std::min(requested_partitions, kMaxPartitions);
    while (partition_count > 1 &&
           points.size() <
               kMinPartitionPoints * static_cast<size_t>(partition_count)) {
      --partition_count;
    }

    if (partition_count > 1) {
      std::vector<std::pair<size_t, size_t>> ranges(partition_count);
#pragma omp parallel
#pragma omp single
      SplitPartitions(points.data(), 0, points.size(), partition_count, 0,
                      ranges.data());

      // Turn each range into the index list its sub-tree is built from and the
      // box the query prunes with. The ranges are disjoint, so this is a
      // parallel loop over partitions.
      partition_indices_.resize(partition_count);
      boxes_.resize(partition_count);
#pragma omp parallel for schedule(dynamic, 1)
      for (int i = 0; i < partition_count; ++i) {
        const size_t begin = ranges[i].first;
        const size_t end = ranges[i].second;
        pcl::IndicesPtr indices(new pcl::Indices(end - begin));
        PartitionBox& box = boxes_[i];
        for (int axis = 0; axis < 3; ++axis) {
          box.lo[axis] = points[begin].c[axis];
          box.hi[axis] = points[begin].c[axis];
        }
        for (size_t j = begin; j < end; ++j) {
          (*indices)[j - begin] = points[j].index;
          for (int axis = 0; axis < 3; ++axis) {
            const float value = points[j].c[axis];
            if (value < box.lo[axis]) {
              box.lo[axis] = value;
            } else if (value > box.hi[axis]) {
              box.hi[axis] = value;
            }
          }
        }
        partition_indices_[i] = indices;
      }

      // Release the packed copy before the trees allocate their own coordinate
      // arrays: what matters is the peak footprint, not the total.
      std::vector<PartitionPoint>().swap(points);

      trees_.resize(partition_count);
      for (int i = 0; i < partition_count; ++i) {
        trees_[i] =
            std::make_shared<pcl::search::KdTree<pcl::PointXYZ>>();
        // Get sorted results from radius search. True should be the default,
        // but be on the safe side for the case of changing defaults:
        trees_[i]->setSortedResults(true);
      }
      // The point of the whole class: FLANN's own buildIndex is serial, but P
      // independent indices are P independent builds. Each KdTreeFLANN owns its
      // FLANN index, its allocator and its parameter map and touches no shared
      // state. Dynamic scheduling because the partitions are equal in point
      // count but not necessarily in build cost.
#pragma omp parallel for schedule(dynamic, 1)
      for (int i = 0; i < partition_count; ++i) {
        trees_[i]->setInputCloud(cloud, partition_indices_[i]);
      }
      return;
    }
  }

  // Single index over the whole cloud: the original code path, unchanged.
  trees_.resize(1);
  trees_[0] = std::make_shared<pcl::search::KdTree<pcl::PointXYZ>>();
  // Get sorted results from radius search. True should be the default, but be
  // on the safe side for the case of changing defaults:
  trees_[0]->setSortedResults(true);
  trees_[0]->setInputCloud(cloud);
}

}  // namespace

void ComputeCompleteness(const MeshLabMeshInfoVector& scan_infos,
                         const std::vector<PointCloudPtr>& scans,
                         const PointCloudPtr& reconstruction,
                         float voxel_size_inv,
                         // Sorted by increasing tolerance.
                         const std::vector<float>& sorted_tolerances,
                         // Indexed by: [tolerance_index]. Range: [0, 1].
                         std::vector<float>* results,
                         // Indexed by: [tolerance_index][scan_point_index].
                         std::vector<std::vector<bool>>* point_is_complete,
                         NnIndexKind nn_index_kind) {
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

  // One sub-index per thread that will query it, so the otherwise serial FLANN
  // build runs on the whole machine. --nn_index flann asks for a single index
  // instead, which is the original code path; both answer every query with the
  // same float (see ReconstructionNearestIndex).
  int requested_partitions = 1;
  if (nn_index_kind == NnIndexKind::kPartitioned) {
#ifdef _OPENMP
    requested_partitions = omp_get_max_threads();
#endif
  }
  ReconstructionNearestIndex reconstruction_nn_index;
  reconstruction_nn_index.Build(reconstruction, requested_partitions);

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

  // Pass 2 (parallel): the nearest neighbour search. Every iteration reads the
  // nearest neighbour index and writes only integer tallies of its own cells
  // plus its own entry of the per-point output, so the loop is free of ordering
  // constraints.
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
      // which asks the kd-tree a much more expensive question than the answer
      // requires. Only the squared distance is ever read below; the neighbour
      // index is never used. With max_nn = 1 FLANN answers a radius query with a
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
      const float nearest_squared_distance =
          reconstruction_nn_index.NearestSquaredDistance(
              search_point, &knn_indices, &knn_squared_dists);
      if (nearest_squared_distance < maximum_tolerance_squared) {
        // Since a reconstruction point was found within the search radius, this
        // scan point is complete for the maximum tolerance, at least. Find the
        // smallest tolerance for which it is still complete.
        int smallest_complete_tolerance_index = 0;
        for (int tolerance_index = static_cast<int>(tolerances_count) - 2;
             tolerance_index >= 0; --tolerance_index) {
          if (sorted_tolerances_squared[tolerance_index] <
              nearest_squared_distance) {
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
                  point_cell_ids_ptrs[grid_index][scan_point_index]) *
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
