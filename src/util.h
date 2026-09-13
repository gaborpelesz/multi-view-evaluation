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
#include <unordered_map>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudPtr;

// Make std::unordered_map work with generic tuples.
// Source: http://stackoverflow.com/questions/7110301
// Note that this is not standard-conformant.
namespace std {
  namespace {
    // Code from boost
    // Reciprocal of the golden ratio helps spread entropy
    //     and handles duplicates.
    // See Mike Seymour in magic-numbers-in-boosthash-combine:
    //     http://stackoverflow.com/questions/4948780
    
    template <class T>
    inline void hash_combine(std::size_t& seed, T const& v) {
      seed ^= std::hash<T>()(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
    }
    
    // Recursive template code derived from Matthieu M.
    template <class Tuple, size_t Index = std::tuple_size<Tuple>::value - 1>
    struct HashValueImpl {
      static void apply(size_t& seed, Tuple const& tuple) {
        HashValueImpl<Tuple, Index-1>::apply(seed, tuple);
        hash_combine(seed, std::get<Index>(tuple));
      }
    };
    
    template <class Tuple>
    struct HashValueImpl<Tuple,0> {
      static void apply(size_t& seed, Tuple const& tuple) {
        hash_combine(seed, std::get<0>(tuple));
      }
    };
  }
  
  template <typename ... TT>
  struct hash<std::tuple<TT...>> {
    size_t operator()(std::tuple<TT...> const& tt) const {
      size_t seed = 0;
      HashValueImpl<std::tuple<TT...> >::apply(seed, tt);
      return seed;
    }
  };
}

// Integer coordinates of a voxel cell. Replaces the std::tuple<int, int, int>
// key that used to be hashed by the specialization above; the full key is kept
// (not packed lossily into fewer bits), so no coordinate range assumption is
// made and two distinct cells can never be confused.
struct VoxelCellKey {
  int32_t x;
  int32_t y;
  int32_t z;
};

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

  // Number of distinct cells seen so far; also the size of the callers' flat
  // per-cell arrays.
  inline uint32_t size() const { return size_; }

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
