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

// Fast reader for the binary PLY files the ETH3D evaluation is given: a
// binary_little_endian file whose "vertex" element has a fixed-width record and
// stores x, y and z as float32.
//
// This is a pure I/O optimization, and it is also the only PLY reader this
// build carries when MVE_PCL_IO_FALLBACK is OFF, so what it accepts has to be
// what the evaluation is actually fed: plain x/y/z, x/y/z with normals, with
// colour, with per-point scalars, in any property order, with or without
// further elements after the vertex block. The coordinate words are copied
// verbatim -- on disk they are already IEEE-754 float32 in the order of
// pcl::PointXYZ's first three members -- so for every file it accepts the
// resulting cloud is bit-identical to the one pcl::io::loadPLYFile() produces,
// down to width, height, is_dense and the sensor pose.
//
// Declined, because reproducing PCL exactly would mean guessing: ASCII and
// big-endian encodings, coordinates stored as double or as an integer type (PCL
// converts those), list properties inside the vertex element, an `obj_info`
// line, and `camera` or `range_grid` elements carrying data. Where PCL's
// general reader is linked in those fall back to it; where it is not, they are
// reported as an unsupported variant so the caller can say so precisely.

namespace fast_ply {

// Why LoadBinaryXyzPly() declined a file. The distinction matters because only
// kNotFastPathShape describes a file that PCL's general reader could plausibly
// have read, and therefore only that value should send a caller looking for the
// fallback (or, in a build without one, produce a typed failure).
enum class LoadFailure {
  // The file was loaded; nothing was declined.
  kNone,
  // The file could not be opened, its size could not be determined, or it stops
  // inside the vertex block. PCL's reader fails on these too (verified on a
  // truncated file), so a caller should report them as it always has.
  kUnreadable,
  // The file opened and is intact, but it is in a PLY variant this reader does
  // not reproduce exactly (see above). PCL's general reader may well handle it.
  kNotFastPathShape,
};

// Loads `path` into `cloud`. Returns true only if the file matched a supported
// layout and its vertex block was read completely; in that case `cloud` holds
// exactly what pcl::io::loadPLYFile() would have produced. Returns false
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
