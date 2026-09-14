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

#include <string>

#include "util.h"

// Fast reader for the single common case in the ETH3D evaluation: a PLY file
// that is binary_little_endian and contains exactly one element, "vertex",
// whose only properties are float x, float y, float z, in that order.
//
// This is a pure I/O optimization. For a file of that shape the on-disk bytes
// are already IEEE-754 float32 in the same order as pcl::PointXYZ's first three
// members, so the points are copied verbatim and the resulting cloud is
// bit-identical to the one pcl::io::loadPLYFile() produces. Every file that
// does not provably have that exact shape (extra properties such as colour,
// normals or intensity, additional elements such as "face", "camera" or
// "range_grid", ASCII or big-endian encoding, a truncated or over-long data
// block) is rejected here so that the caller falls back to PCL's general
// reader, which stays the reference implementation for all other inputs.
namespace fast_ply {

// Why LoadBinaryXyzPly() declined a file. The distinction matters because only
// kNotFastPathShape describes a file that PCL's general reader could plausibly
// have read, and therefore only that value should send a caller looking for the
// fallback (or, in a build without one, produce a typed failure).
enum class LoadFailure {
  // The file was loaded; nothing was declined.
  kNone,
  // The file could not be opened, or its size could not be determined. PCL's
  // reader would not have got any further.
  kCannotOpen,
  // The file opened, but it is not a plain binary_little_endian float x/y/z PLY
  // with a single vertex element -- or its data block does not have exactly the
  // length the header implies. PCL's general reader may well handle it.
  kNotFastPathShape,
};

// Loads `path` into `cloud` on the fast path. Returns true only if the file
// matched the supported layout and was read completely; in that case `cloud`
// holds exactly what pcl::io::loadPLYFile() would have produced. Returns false
// without leaving usable data in `cloud` otherwise; the caller must then fall
// back to pcl::io::loadPLYFile() (which will also emit the usual PCL error
// messages if the file is genuinely unreadable).
// When `failure` is not null it receives the reason the file was declined, or
// LoadFailure::kNone on success.
bool LoadBinaryXyzPly(const std::string& path, PointCloud* cloud,
                      LoadFailure* failure = nullptr);

// Writes `cloud` to `path` as a binary_little_endian PLY, producing byte for
// byte the same file that pcl::io::savePLYFileBinary() produces for a
// pcl::PointCloud<pcl::PointXYZRGB>.
//
// This exists only so that the evaluation binary does not have to link
// libpcl_io. See the implementation for the layout this reproduces and for how
// each part of it was pinned down against PCL's own output. Returns false if
// the file could not be created or not written in full; as in PCL, a cloud with
// no points is refused and no file is created.
bool WriteBinaryXyzRgbPly(const std::string& path,
                          const pcl::PointCloud<pcl::PointXYZRGB>& cloud);

}  // namespace fast_ply
