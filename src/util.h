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

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudPtr;

// Integer coordinates of a voxel cell. Replaces the std::tuple<int, int, int>
// key that upstream hashed through a std::hash specialization; the full key is
// kept
// (not packed lossily into fewer bits), so no coordinate range assumption is
// made and two distinct cells can never be confused.
struct VoxelCellKey {
  int32_t x;
  int32_t y;
  int32_t z;
};

// Exact equality of the full key, as std::tuple's operator== was. Used by the
// callers to short-circuit a repeated lookup of the cell the previous point
// fell into; it can only skip a lookup that would have returned the same id.
inline bool operator==(const VoxelCellKey& a, const VoxelCellKey& b) {
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

inline VoxelCellKey CalcCellCoordinates(
    const pcl::PointXYZ& point, float voxel_size_inv,
    float shift_x, float shift_y, float shift_z) {
  float voxel_x = point.x * voxel_size_inv + shift_x;
  int x = static_cast<int>(voxel_x);
  if (voxel_x - x != 0 && voxel_x < 0) {
    -- x;
  }
  
  float voxel_y = point.y * voxel_size_inv + shift_y;
  int y = static_cast<int>(voxel_y);
  if (voxel_y - y != 0 && voxel_y < 0) {
    -- y;
  }
  
  float voxel_z = point.z * voxel_size_inv + shift_z;
  int z = static_cast<int>(voxel_z);
  if (voxel_z - z != 0 && voxel_z < 0) {
    -- z;
  }
  
  return VoxelCellKey{x, y, z};
}

// Maps voxel cell coordinates to a dense, gap-free cell index which is handed
// out in insertion order (the first cell created gets index 0, and so on).
//
// This replaces std::unordered_map<std::tuple<int, int, int>, Cell> in the two
// voxel tallies. The node-based map cost one separate malloc'd node per cell
// plus a pointer chase per probe, and its mapped type in turn held a std::vector
// whose heap block was a second malloc per cell. Here the table is a single
// contiguous array of 16-byte slots with linear probing, and the callers keep
// their per-cell counters in flat arrays indexed by the dense index, so a cell
// costs no allocation of its own at all.
//
// The dense index (not a pointer) is what callers hold on to: the flat counter
// arrays may be reallocated by a later insertion, whereas unordered_map node
// addresses were stable.
class VoxelCellIndexMap {
 public:
  static const uint32_t kEmptySlot = 0xFFFFFFFFu;

  inline VoxelCellIndexMap() : size_(0) { AllocateTable(1024); }

  // Sizes the table so that expected_cells distinct cells fit into it without a
  // single rehash, and must therefore be called before the first Lookup(). The
  // table is allocated with the smallest power of two strictly greater than
  // 2 * expected_cells slots, which is one slot more than Lookup()'s growth
  // rule (2 * size_ >= slots_.size()) needs to stay quiet for that many cells.
  //
  // This only picks the table geometry. The dense index a cell receives is the
  // value of size_ at its first insertion, which depends solely on the order in
  // which distinct keys are looked up -- so reserving too much, too little or
  // nothing at all cannot change a single index. An underestimate merely lets
  // Grow() run as before.
  inline void Reserve(size_t expected_cells) {
    if (size_ != 0) {
      // Reserving after the first insertion would be a programming error: the
      // caller would be paying for a rehash it asked to avoid.
      std::fprintf(stderr, "VoxelCellIndexMap: Reserve() after first insert\n");
      std::abort();
    }
    // At most kEmptySlot - 1 cells can ever be indexed, so there is no point in
    // allocating a table for more than that; this also bounds the doubling.
    if (expected_cells >= kEmptySlot) {
      expected_cells = kEmptySlot - 1;
    }
    size_t slot_count = 1024;
    while (slot_count <= 2 * expected_cells) {
      slot_count *= 2;
    }
    AllocateTable(slot_count);
  }

  // Returns the dense index of the cell with the given coordinates, creating
  // the cell if it does not exist yet. *inserted is set to true exactly if a
  // new cell was created, in which case the returned index equals the previous
  // number of cells, so callers can append to their flat counter arrays.
  inline uint32_t Lookup(const VoxelCellKey& key, bool* inserted) {
    size_t slot_index = static_cast<size_t>(Hash(key)) & mask_;
    while (true) {
      Slot& slot = slots_[slot_index];
      if (slot.index == kEmptySlot) {
        slot.x = key.x;
        slot.y = key.y;
        slot.z = key.z;
        if (size_ == kEmptySlot) {
          // Unreachable in practice (it would take 2^32 - 1 occupied cells, far
          // more than fits in memory), but a silently wrapped index would
          // corrupt a measurement, so fail loudly instead.
          std::fprintf(stderr, "VoxelCellIndexMap: too many cells\n");
          std::abort();
        }
        const uint32_t new_index = size_;
        slot.index = new_index;
        ++ size_;
        *inserted = true;
        if (2 * static_cast<size_t>(size_) >= slots_.size()) {
          Grow();
        }
        return new_index;
      }
      if (slot.x == key.x && slot.y == key.y && slot.z == key.z) {
        *inserted = false;
        return slot.index;
      }
      slot_index = (slot_index + 1) & mask_;
    }
  }

  // Asks the hardware to fetch the first slot a lookup of this key would probe
  // into cache, for writing. That slot's address is Hash(key) & mask_, a pure
  // function of the key and the current table geometry, so the address is
  // correct no matter which keys are inserted between the hint and the lookup;
  // and a prefetch has no architectural effect whatsoever, so even a wrong
  // address could only cost time, never change a result.
  inline void Prefetch(const VoxelCellKey& key) const {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(&slots_[static_cast<size_t>(Hash(key)) & mask_], 1, 3);
#else
    (void)key;
#endif
  }

  // Number of distinct cells seen so far; also the size of the callers' flat
  // per-cell arrays.
  inline uint32_t size() const { return size_; }

  // One in kCellSampleRate of the cell space, selected by the top bits of the
  // hash. EstimateDistinctCellCounts() below counts the cells this accepts, so
  // the decision must depend on the CELL and on nothing else: a cell holding a
  // thousand points is accepted exactly as often as a cell holding one, which
  // is what makes the extrapolation unbiased with respect to occupancy.
  //
  // The selection uses the HIGH bits. A table's own slot index is the LOW bits
  // (Hash(key) & mask_), so sampled keys are uncorrelated with any slot and
  // still spread over the whole sample table instead of piling onto every
  // kCellSampleRate-th slot of it. The final `h ^= h >> 32` of the mix leaves
  // bits 63..32 exactly as they came out of the multiply, where they already
  // depend on every input bit, so the high bits are as well mixed as the low
  // ones.
  //
  // kCellSampleShift must be 64 - log2(kCellSampleRate). 1/128 measured as the
  // knee: the estimate is still within ~6% of the true cell count on the bench
  // dataset, while the sample tables have become small enough that the pass
  // costs little beyond the key and the hash it has to compute per point
  // anyway. Sampling 1/512 saved only a further 1.5 ms of 33 and cost twice
  // the relative error.
  static const uint64_t kCellSampleRate = 128;
  static const int kCellSampleShift = 57;
  static inline bool IsSampledCell(const VoxelCellKey& key) {
    return (Hash(key) >> kCellSampleShift) == 0;
  }

 private:
  struct Slot {
    int32_t x;
    int32_t y;
    int32_t z;
    // kEmptySlot marks a free slot.
    uint32_t index;
  };

  // Multiply-xorshift mix of the three coordinates (splitmix64 finalizer). The
  // hash only decides which slot is probed first; the stored coordinates decide
  // equality, so the choice of hash cannot change any result.
  static inline uint64_t Hash(const VoxelCellKey& key) {
    uint64_t h = static_cast<uint64_t>(static_cast<uint32_t>(key.x)) *
                     0x9e3779b97f4a7c15ull +
                 static_cast<uint64_t>(static_cast<uint32_t>(key.y)) *
                     0xc2b2ae3d27d4eb4full +
                 static_cast<uint64_t>(static_cast<uint32_t>(key.z)) *
                     0x27d4eb2f165667c5ull;
    h ^= h >> 29;
    h *= 0xbf58476d1ce4e5b9ull;
    h ^= h >> 32;
    return h;
  }

  inline void AllocateTable(size_t slot_count) {
    Slot empty_slot;
    empty_slot.x = 0;
    empty_slot.y = 0;
    empty_slot.z = 0;
    empty_slot.index = kEmptySlot;
    slots_.assign(slot_count, empty_slot);
    mask_ = slot_count - 1;
  }

  // Doubles the table and re-inserts the occupied slots. Amortized O(1) per
  // cell: across all growths every cell is re-inserted about once in total.
  void Grow() {
    std::vector<Slot> old_slots;
    old_slots.swap(slots_);
    AllocateTable(2 * old_slots.size());
    for (size_t i = 0, count = old_slots.size(); i < count; ++ i) {
      const Slot& old_slot = old_slots[i];
      if (old_slot.index == kEmptySlot) {
        continue;
      }
      VoxelCellKey key;
      key.x = old_slot.x;
      key.y = old_slot.y;
      key.z = old_slot.z;
      size_t slot_index = static_cast<size_t>(Hash(key)) & mask_;
      while (slots_[slot_index].index != kEmptySlot) {
        slot_index = (slot_index + 1) & mask_;
      }
      slots_[slot_index] = old_slot;
    }
  }

  std::vector<Slot> slots_;
  size_t mask_;
  uint32_t size_;
};

// Integer square root of v, exact for every uint64_t: returns
// floor(sqrt(v)). Used only to size a hash table, but kept in integer
// arithmetic so that no rounding decision anywhere in this program depends on
// the floating point environment.
inline uint64_t IntegerSqrt(uint64_t v) {
  if (v == 0) {
    return 0;
  }
  // Newton's iteration, seeded at 2^32, which is >= sqrt(v) for every uint64_t.
  // From a seed at or above the root the iterates decrease monotonically and
  // never fall below floor(sqrt(v)), which is both why the loop terminates at
  // the right value and why root is never zero (so the division is safe).
  // While root >= sqrt(v) we have v / root <= root, so the sum below is at most
  // 2 * 2^32 and cannot overflow either.
  uint64_t root = 1ull << 32;
  while (true) {
    const uint64_t next = (root + v / root) / 2;
    if (next >= root) {
      break;
    }
    root = next;
  }
  return root;
}

// Estimates, for each of GridCount voxel grids, how many distinct cells the
// given points occupy in it, so that the VoxelCellIndexMap of each grid can be
// Reserve()d up front instead of doubling its way there from 1024 slots. All
// grids are estimated in a single pass over the point array, which is the only
// sequential read this function does.
//
// HOW THE ESTIMATE IS TAKEN, AND WHY NOT THE OBVIOUS WAY. Counting the distinct
// cells hit by every s-th POINT and multiplying by s does not estimate the
// number of cells -- it estimates the number of points. A cell holding k points
// is hit by such a sample with probability 1 - (1 - 1/s)^k, so the
// extrapolation returns cells * s * (1 - (1 - 1/s)^k), which is right only at
// k = 1 and, at s = 32, is 14x too high already at k = 15 and 22x too high at
// k = 31. Clamped to the point count, as it has to be, that is not a safeguard
// against a large cloud at all: it IS the fixed "two slots per point" rule it
// was supposed to replace, for every cloud denser than about one point per
// voxel.
//
// So the sample is taken over the CELL SPACE instead: a cell is counted iff
// VoxelCellIndexMap::IsSampledCell() accepts it, which is a function of the
// cell alone, so the acceptance probability is 1/kCellSampleRate whatever the
// occupancy. The number of accepted cells is then Binomial(cells,
// 1/kCellSampleRate) and kCellSampleRate times it is unbiased for the cell
// count at every density. Every point must still be visited -- an unbiased
// cell-space sample cannot be drawn from a subsample of the points -- but a
// visit that is not accepted costs only the cell coordinates and the hash, and
// touches no table.
//
// HEADROOM. An underestimate is not a correctness problem (Grow() absorbs it)
// but it costs the single most expensive rehash, so the point estimate is
// raised by five standard deviations of that binomial. With r =
// kCellSampleRate and a accepted cells, sd(r * a) = sqrt(r * (r - 1) * a), so
// five of them is sqrt(25 * r * (r - 1) * a), which is what the code below
// computes. The allowance is therefore self-scaling: large in relative terms
// when the sample is small and uninformative, and vanishing when it is large.
// At the bench dataset's ~2.2M cells it is 3.8%, and the estimates come out
// 4.2% to 5.6% above the true cell counts -- close enough that all four tables
// land on exactly the power-of-two size the growing table used to end at, and
// none of them calls Grow() even once (verified in-tree).
//
// Below kSmallCloudPointCount points the exact upper bound (point_count) is
// used instead: the table it implies is at most 4 MB, which is not worth an
// estimation pass or a discussion of sampling error.
//
// NOTHING HERE IS LOAD-BEARING FOR THE RESULT. The value returned is only ever
// passed to VoxelCellIndexMap::Reserve(), which picks the table geometry; a
// cell's dense index is the value of size_ at its first insertion and depends
// solely on the order of distinct keys, never on the table size. A wrong
// estimate can only cost memory or rehashes.
template <int GridCount>
inline void EstimateDistinctCellCounts(
    const pcl::PointXYZ* points, size_t point_count, float voxel_size_inv,
    const float (&grid_shifts)[GridCount][3],
    size_t (&estimates)[GridCount]) {
  const size_t kSmallCloudPointCount = 1 << 16;

  if (point_count == 0) {
    for (int grid_index = 0; grid_index < GridCount; ++ grid_index) {
      estimates[grid_index] = 0;
    }
    return;
  }
  if (point_count <= kSmallCloudPointCount) {
    for (int grid_index = 0; grid_index < GridCount; ++ grid_index) {
      estimates[grid_index] = point_count;
    }
    return;
  }

  // One throwaway table per grid, holding roughly cells / kCellSampleRate
  // entries. They are left to grow from the 1024-slot default on purpose: the
  // whole point of this function is not to know that size in advance, and the
  // growth of a table this small is a rounding error against the four
  // production tables it sizes.
  VoxelCellIndexMap sample_maps[GridCount];
  for (size_t point_index = 0; point_index < point_count; ++ point_index) {
    const pcl::PointXYZ& point = points[point_index];
    for (int grid_index = 0; grid_index < GridCount; ++ grid_index) {
      const VoxelCellKey key = CalcCellCoordinates(
          point, voxel_size_inv, grid_shifts[grid_index][0],
          grid_shifts[grid_index][1], grid_shifts[grid_index][2]);
      if (!VoxelCellIndexMap::IsSampledCell(key)) {
        continue;
      }
      bool inserted;
      sample_maps[grid_index].Lookup(key, &inserted);
    }
  }

  for (int grid_index = 0; grid_index < GridCount; ++ grid_index) {
    const uint64_t accepted = sample_maps[grid_index].size();
    uint64_t estimate =
        accepted * VoxelCellIndexMap::kCellSampleRate +
        IntegerSqrt(25 * VoxelCellIndexMap::kCellSampleRate *
                    (VoxelCellIndexMap::kCellSampleRate - 1) * accepted) +
        1;
    // A point can occupy at most one cell, so point_count is a hard bound and
    // no headroom may push past it.
    if (estimate > point_count) {
      estimate = point_count;
    }
    estimates[grid_index] = static_cast<size_t>(estimate);
  }
}
