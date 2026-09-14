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

#include "fast_ply.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <boost/filesystem.hpp>

namespace fast_ply {
namespace {

// Bytes per vertex on disk for the supported layout (float x, y, z).
constexpr std::uint64_t kDiskPointSize = 3 * sizeof(float);
// Number of vertices staged per read() call. 8192 vertices are 96 KiB, which
// stays resident in cache between the read and the stride expansion.
constexpr std::size_t kChunkPoints = 8192;
// A PLY header longer than this is not worth special-casing; fall back.
constexpr std::size_t kMaxHeaderBytes = 64 * 1024;
// IEEE-754 binary32 exponent field; all ones means infinity or NaN.
constexpr std::uint32_t kFloatExponentMask = 0x7f800000u;

bool HostIsLittleEndian() {
  const std::uint32_t value = 1;
  unsigned char bytes[sizeof(value)];
  std::memcpy(bytes, &value, sizeof(value));
  return bytes[0] == 1;
}

// Splits a header line at runs of spaces and tabs.
std::vector<std::string> Tokenize(const std::string& line) {
  std::vector<std::string> tokens;
  std::size_t pos = 0;
  while (pos < line.size()) {
    while (pos < line.size() && (line[pos] == ' ' || line[pos] == '\t')) {
      ++pos;
    }
    const std::size_t start = pos;
    while (pos < line.size() && line[pos] != ' ' && line[pos] != '\t') {
      ++pos;
    }
    if (pos > start) {
      tokens.push_back(line.substr(start, pos - start));
    }
  }
  return tokens;
}

// Parses a non-negative decimal integer. Returns false on anything else,
// including empty strings, signs, overflow and trailing characters.
bool ParseCount(const std::string& text, std::uint64_t* value) {
  if (text.empty() || text.size() > 20) {
    return false;
  }
  std::uint64_t result = 0;
  for (const char c : text) {
    if (c < '0' || c > '9') {
      return false;
    }
    const std::uint64_t digit = static_cast<std::uint64_t>(c - '0');
    if (result > (UINT64_MAX - digit) / 10) {
      return false;
    }
    result = 10 * result + digit;
  }
  *value = result;
  return true;
}

bool IsFloat32Type(const std::string& type) {
  return type == "float" || type == "float32";
}

// Parses the header of an already-opened PLY file. On success, reports the
// vertex count and the byte offset at which the vertex data starts. Returns
// false for every header that is not provably "binary_little_endian 1.0 with a
// single vertex element carrying exactly float x, float y, float z".
bool ParseSupportedHeader(const std::vector<char>& buffer,
                          std::uint64_t* vertex_count,
                          std::uint64_t* data_offset) {
  bool seen_magic = false;
  bool seen_format = false;
  bool seen_vertex_element = false;
  bool seen_end_header = false;
  int property_count = 0;
  std::uint64_t count = 0;

  static const char* const kExpectedProperties[3] = {"x", "y", "z"};

  std::size_t pos = 0;
  while (pos < buffer.size()) {
    // Isolate one line. A header line that is not newline-terminated inside the
    // buffer means the header is longer than what was read: give up.
    const void* newline = std::memchr(buffer.data() + pos, '\n',
                                      buffer.size() - pos);
    if (newline == nullptr) {
      return false;
    }
    const std::size_t line_end =
        static_cast<std::size_t>(static_cast<const char*>(newline) -
                                 buffer.data());
    std::size_t content_end = line_end;
    if (content_end > pos && buffer[content_end - 1] == '\r') {
      --content_end;
    }
    const std::string line(buffer.data() + pos, content_end - pos);
    pos = line_end + 1;

    const std::vector<std::string> tokens = Tokenize(line);
    if (tokens.empty()) {
      // The PLY header grammar has no blank lines; do not guess.
      return false;
    }
    const std::string& keyword = tokens[0];

    if (!seen_magic) {
      if (keyword != "ply" || tokens.size() != 1) {
        return false;
      }
      seen_magic = true;
      continue;
    }

    // `obj_info` is NOT free-form: PCL's PLY reader maps obj_info num_cols /
    // num_rows onto the cloud width and height, and echo_rgb_offset_x|y|z onto
    // the sensor origin. A file carrying it would load with a different point
    // count here than under pcl::io::loadPLYFile, so refuse the fast path.
    if (keyword == "obj_info") {
      return false;
    }
    if (keyword == "comment") {
      // Free-form text, carries no data; ignoring it is safe.
      continue;
    }

    if (keyword == "format") {
      if (seen_format || tokens.size() != 3 ||
          tokens[1] != "binary_little_endian" || tokens[2] != "1.0") {
        return false;
      }
      seen_format = true;
      continue;
    }

    if (keyword == "element") {
      // Exactly one element is supported, and it must be the vertex element.
      // Anything else (face, camera, range_grid, ...) changes what PCL's
      // reader produces, so it must go through PCL.
      if (seen_vertex_element || tokens.size() != 3 || tokens[1] != "vertex" ||
          !ParseCount(tokens[2], &count)) {
        return false;
      }
      seen_vertex_element = true;
      continue;
    }

    if (keyword == "property") {
      if (!seen_vertex_element || property_count >= 3 || tokens.size() != 3 ||
          !IsFloat32Type(tokens[1]) ||
          tokens[2] != kExpectedProperties[property_count]) {
        return false;
      }
      ++property_count;
      continue;
    }

    if (keyword == "end_header") {
      if (tokens.size() != 1) {
        return false;
      }
      seen_end_header = true;
      break;
    }

    // Unknown keyword: fall back rather than guess at its meaning.
    return false;
  }

  if (!seen_end_header || !seen_format || !seen_vertex_element ||
      property_count != 3) {
    return false;
  }
  // pcl::PCLPointCloud2::width is 32 bit, and the byte count below must not
  // overflow. No real file comes close to either bound; fall back if one does.
  if (count > UINT32_MAX || count > UINT64_MAX / kDiskPointSize) {
    return false;
  }

  *vertex_count = count;
  *data_offset = pos;
  return true;
}

}  // namespace

bool LoadBinaryXyzPly(const std::string& path, PointCloud* cloud) {
  if (!HostIsLittleEndian()) {
    return false;
  }

  std::FILE* file = std::fopen(path.c_str(), "rb");
  if (file == nullptr) {
    return false;
  }

  std::vector<char> header(kMaxHeaderBytes);
  const std::size_t header_bytes =
      std::fread(header.data(), 1, header.size(), file);
  header.resize(header_bytes);

  std::uint64_t vertex_count = 0;
  std::uint64_t data_offset = 0;
  if (!ParseSupportedHeader(header, &vertex_count, &data_offset)) {
    std::fclose(file);
    return false;
  }

  // The data block must be exactly as long as the header claims. A short file
  // is corrupt; a long one holds something this reader did not account for.
  // Checking this before allocating also keeps a corrupt vertex count from
  // turning into a huge allocation.
  boost::system::error_code error;
  const boost::uintmax_t file_size = boost::filesystem::file_size(path, error);
  if (error || static_cast<std::uint64_t>(file_size) !=
                   data_offset + vertex_count * kDiskPointSize) {
    std::fclose(file);
    return false;
  }
  // The header is bounded by kMaxHeaderBytes, so this offset always fits.
  if (std::fseek(file, static_cast<long>(data_offset), SEEK_SET) != 0) {
    std::fclose(file);
    return false;
  }

  // Match what pcl::io::loadPLYFile() leaves behind for a file without a
  // "camera" element: an unorganized cloud with an identity sensor pose.
  // resize() default-constructs every point, which is what sets the fourth
  // (padding) float of pcl::PointXYZ to 1.0f, exactly as PCL's conversion from
  // pcl::PCLPointCloud2 does; only x, y and z are written below.
  cloud->clear();
  cloud->sensor_origin_ = Eigen::Vector4f::Zero();
  cloud->sensor_orientation_ = Eigen::Quaternionf::Identity();
  cloud->resize(static_cast<std::size_t>(vertex_count));
  cloud->width = static_cast<std::uint32_t>(vertex_count);
  cloud->height = 1;

  std::vector<float> staging(3 * kChunkPoints);
  // PCL's PLY reader clears is_dense as soon as one vertex coordinate is not
  // finite, and is_dense selects different code paths further downstream (the
  // non-dense branch of pcl::transformPointCloud, the invalid-point filtering
  // in the FLANN k-d tree), so the same flag has to be derived here. The test
  // is done on the raw bit pattern: a float32 is non-finite exactly when its
  // exponent field is all ones, which is what std::isfinite() reports.
  std::uint32_t any_non_finite = 0;
  pcl::PointXYZ* dest = cloud->points.data();
  std::uint64_t remaining = vertex_count;
  while (remaining > 0) {
    const std::size_t chunk = static_cast<std::size_t>(
        remaining < kChunkPoints ? remaining : kChunkPoints);
    if (std::fread(staging.data(), static_cast<std::size_t>(kDiskPointSize),
                   chunk, file) != chunk) {
      std::fclose(file);
      cloud->clear();
      return false;
    }

    const std::size_t word_count = 3 * chunk;
    for (std::size_t i = 0; i < word_count; ++i) {
      std::uint32_t bits;
      std::memcpy(&bits, staging.data() + i, sizeof(bits));
      any_non_finite |= static_cast<std::uint32_t>(
          (bits & kFloatExponentMask) == kFloatExponentMask);
    }

    const float* source = staging.data();
    for (std::size_t i = 0; i < chunk; ++i, source += 3, ++dest) {
      // Verbatim copy of the three float32 values; the padding float keeps the
      // 1.0f that default construction wrote.
      std::memcpy(dest->data, source, kDiskPointSize);
    }
    remaining -= chunk;
  }
  cloud->is_dense = (any_non_finite == 0);

  std::fclose(file);
  return true;
}

// ---------------------------------------------------------------------------
// Writer.
//
// The layout below is not a guess: it was read back out of files produced by
// pcl::io::savePLYFileBinary() on this exact PCL (1.15), both from this
// program's own classification clouds and from a probe cloud carrying a
// deliberately asymmetric sensor pose, a NaN, an infinity and a negative zero.
// Every constant here is what that output contained.
//
//   ply
//   format binary_little_endian 1.0
//   comment PCL generated
//   element vertex <width * height>
//   property float x / y / z
//   property uchar red / green / blue      <- PCL expands the packed "rgb"
//   element face 0                            field into three uchars
//   element camera 1
//   property float view_px / view_py / view_pz
//   property float x_axisx ... z_axisz
//   property float focal / scalex / scaley / centerx / centery
//   property int viewportx / viewporty
//   property float k1 / k2
//   end_header
//
// then width*height vertex records of 15 bytes (three little-endian float32
// then three uchar), then nothing for the empty face element, then one 84-byte
// camera record. The camera record holds the cloud's sensor pose: the origin,
// the rotation matrix in ROW-MAJOR order (x_axis is row 0, verified with a
// 90-degree rotation about Z, which is asymmetric enough to tell the two
// conventions apart), five zero floats for the intrinsics PCL has no value for,
// the cloud width and height as int32, and two zero floats for the distortion
// coefficients.
//
// The vertex payload is a verbatim copy of the point's first three float32
// words: PCL's writer memcpy's the field out of the PCLPointCloud2 blob and
// writes it unchanged, so non-finite coordinates are preserved bit for bit and
// are NOT filtered (confirmed on the probe cloud). That matters, because the
// classification clouds inherit whatever the input PLY contained.
// ---------------------------------------------------------------------------

namespace {

// Bytes per vertex on disk for the written layout (float x, y, z + uchar rgb).
constexpr std::size_t kRgbDiskPointSize = 3 * sizeof(float) + 3;
// Bytes in the trailing camera record: 17 float32, 2 int32, 2 float32.
constexpr std::size_t kCameraRecordSize = 17 * 4 + 2 * 4 + 2 * 4;
// Vertices staged per fwrite() call; 8192 records are 120 KiB.
constexpr std::size_t kWriteChunkPoints = 8192;

// Appends the little-endian representation of one 4-byte value. The host is
// checked to be little-endian by the caller, so this is a plain copy.
template <typename T>
void AppendLe32(std::vector<char>* out, T value) {
  static_assert(sizeof(T) == 4, "only 4-byte scalars are written here");
  char bytes[4];
  std::memcpy(bytes, &value, sizeof(bytes));
  out->insert(out->end(), bytes, bytes + sizeof(bytes));
}

}  // namespace

bool WriteBinaryXyzRgbPly(const std::string& path,
                          const pcl::PointCloud<pcl::PointXYZRGB>& cloud) {
  if (!HostIsLittleEndian()) {
    return false;
  }

  // PCL counts the vertices as width * height, not as points.size(); the two
  // agree for every cloud this program writes, but the file has to say what
  // PCL's would say.
  const std::uint64_t vertex_count =
      static_cast<std::uint64_t>(cloud.width) * cloud.height;
  if (vertex_count == 0 || vertex_count > cloud.points.size()) {
    // pcl::PLYWriter::writeBinary() refuses an empty cloud with
    // "Input point cloud has no data!" and creates no file at all; not
    // creating one here keeps that behaviour. The second half of the test is a
    // bounds guard, not a PCL behaviour: it cannot trigger for a cloud built by
    // resize().
    return false;
  }

  std::FILE* file = std::fopen(path.c_str(), "wb");
  if (file == nullptr) {
    return false;
  }

  // The header text is fixed apart from the vertex count.
  std::string header =
      "ply\n"
      "format binary_little_endian 1.0\n"
      "comment PCL generated\n"
      "element vertex ";
  header += std::to_string(vertex_count);
  header +=
      "\n"
      "property float x\n"
      "property float y\n"
      "property float z\n"
      "property uchar red\n"
      "property uchar green\n"
      "property uchar blue\n"
      "element face 0\n"
      "element camera 1\n"
      "property float view_px\n"
      "property float view_py\n"
      "property float view_pz\n"
      "property float x_axisx\n"
      "property float x_axisy\n"
      "property float x_axisz\n"
      "property float y_axisx\n"
      "property float y_axisy\n"
      "property float y_axisz\n"
      "property float z_axisx\n"
      "property float z_axisy\n"
      "property float z_axisz\n"
      "property float focal\n"
      "property float scalex\n"
      "property float scaley\n"
      "property float centerx\n"
      "property float centery\n"
      "property int viewportx\n"
      "property int viewporty\n"
      "property float k1\n"
      "property float k2\n"
      "end_header\n";
  if (std::fwrite(header.data(), 1, header.size(), file) != header.size()) {
    std::fclose(file);
    return false;
  }

  std::vector<char> staging(kWriteChunkPoints * kRgbDiskPointSize);
  const pcl::PointXYZRGB* source = cloud.points.data();
  std::uint64_t remaining = vertex_count;
  while (remaining > 0) {
    const std::size_t chunk = static_cast<std::size_t>(
        remaining < kWriteChunkPoints ? remaining : kWriteChunkPoints);
    char* dest = staging.data();
    for (std::size_t i = 0; i < chunk; ++i, ++source, dest += kRgbDiskPointSize) {
      // Verbatim copy of the three float32 coordinate words, exactly as PCL
      // does it: no rounding, no finiteness test, no reordering.
      std::memcpy(dest, source->data, 3 * sizeof(float));
      // PCL unpacks the "rgb" field into three uchars in the order the header
      // declares them. PointXYZRGB stores the channels as b, g, r, a in
      // memory, so naming the members rather than copying bytes is what keeps
      // the order right.
      dest[12] = static_cast<char>(source->r);
      dest[13] = static_cast<char>(source->g);
      dest[14] = static_cast<char>(source->b);
    }
    const std::size_t bytes = chunk * kRgbDiskPointSize;
    if (std::fwrite(staging.data(), 1, bytes, file) != bytes) {
      std::fclose(file);
      return false;
    }
    remaining -= chunk;
  }

  // The "face" element is declared with a count of zero, so nothing follows it.
  // The "camera" element is one record describing the cloud's sensor pose.
  const Eigen::Matrix3f rotation = cloud.sensor_orientation_.toRotationMatrix();
  std::vector<char> camera;
  camera.reserve(kCameraRecordSize);
  for (int i = 0; i < 3; ++i) {
    AppendLe32(&camera, cloud.sensor_origin_[i]);
  }
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      AppendLe32(&camera, rotation(row, column));
    }
  }
  // focal, scalex, scaley, centerx, centery: PCL has no camera intrinsics to
  // put here and writes five zeros.
  for (int i = 0; i < 5; ++i) {
    AppendLe32(&camera, 0.0f);
  }
  AppendLe32(&camera, static_cast<std::int32_t>(cloud.width));
  AppendLe32(&camera, static_cast<std::int32_t>(cloud.height));
  // k1, k2: the distortion coefficients, likewise always zero.
  AppendLe32(&camera, 0.0f);
  AppendLe32(&camera, 0.0f);
  if (camera.size() != kCameraRecordSize ||
      std::fwrite(camera.data(), 1, camera.size(), file) != camera.size()) {
    std::fclose(file);
    return false;
  }

  // A write error can still surface here, when the last buffer is flushed.
  return std::fclose(file) == 0;
}

}  // namespace fast_ply
