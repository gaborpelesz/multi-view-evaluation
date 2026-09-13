cmake_minimum_required(VERSION 3.16)
project(ETH3DMultiViewEvaluation CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Settings. RelWithDebInfo is the canonical benchmark configuration: it matches
# what benchmark/eth3d/multi-view-evaluation-compat.patch produces and keeps
# frame pointers/line tables for the profiler.
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type not specified, using RelWithDebInfo")
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# ---------------------------------------------------------------------------
# Optimization flags.
#
# The stock RelWithDebInfo configuration is "-O2 -g -DNDEBUG" with no
# architecture targeting and no interprocedural optimization. This raises it to
# -O3, targets the build host's own ISA, and enables ThinLTO. -g is kept so the
# profiler still resolves symbols and line numbers, and -DNDEBUG is kept because
# dropping it would re-enable PCL's and Eigen's assertions in the hot loops.
#
# BYTE-EXACTNESS
# --------------
# None of the flags below relaxes IEEE-754 semantics. -ffast-math, -Ofast,
# -funsafe-math-optimizations, -ffinite-math-only, -fno-signed-zeros,
# -freciprocal-math and anything enabling flush-to-zero / denormals-are-zero are
# deliberately NOT used: they would let the compiler reassociate and approximate
# the float arithmetic that decides every point's classification. -ffp-contract
# is likewise left at the compiler default, so FMA contraction behaves exactly
# as it does in the stock build -- see the caveat below, it matters.
#
# -O3, -mcpu=native and ThinLTO were each verified byte-identical to the -O2
# baseline by the harness gate (four stdout result lines plus SHA-256 over
# twelve per-point classification clouds, on the tiny and bench datasets, at 1
# and 12 threads) on arm64 / Apple clang 21 / apple-m4.
#
# MEASURED CAVEAT -- READ BEFORE ENABLING MVE_NATIVE_ARCH ON x86-64
# -----------------------------------------------------------------
# This program is NOT invariant to FMA contraction. Rebuilding these exact
# sources with -ffp-contract=off and nothing else changed moved the printed
# F1-scores and changed all twelve cloud hashes: contraction perturbs the last
# bits of the Eigen transforms, which flips the classification of the points
# sitting exactly on a tolerance boundary.
#
# On arm64 that is harmless here, because the baseline compiler already targets
# an FMA-capable core (apple-m1 by default): -mcpu=native only moves the
# scheduling model up to apple-m4 and adds ISA features the compiler does not
# auto-select for this code, so no new FMA rounding is introduced. That is why
# the gate stays green.
#
# On x86-64 it is NOT harmless. The x86 baseline ISA is plain SSE2 with no FMA;
# -march=native on an AVX2/FMA Ryzen introduces FMA -- both through contraction
# and through Eigen's explicit EIGEN_VECTORIZE_FMA kernels -- and widens the
# packets Eigen reduces over. Expect the printed values to move there. Re-run
# the gate on that host before trusting any number it produces, and if the gate
# fails, configure with -DMVE_NATIVE_ARCH=OFF. Never regenerate the golden to
# make it pass.
# ---------------------------------------------------------------------------
option(MVE_NATIVE_ARCH
       "Target the build host's own ISA (-mcpu=native / -march=native). Must be re-gated on any host whose baseline ISA lacks FMA."
       ON)
option(MVE_LTO "Enable interprocedural optimization (ThinLTO)." ON)

# -O2 -> -O3, keeping -g and -DNDEBUG exactly as the toolchain file set them.
string(REPLACE "-O2" "-O3"
       CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")

set(MVE_ARCH_FLAGS "")
if(MVE_NATIVE_ARCH AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  include(CheckCXXCompilerFlag)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64|ARM64)$")
    check_cxx_compiler_flag("-mcpu=native" MVE_HAVE_MCPU_NATIVE)
    if(MVE_HAVE_MCPU_NATIVE)
      set(MVE_ARCH_FLAGS "-mcpu=native")
    endif()
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    check_cxx_compiler_flag("-march=native" MVE_HAVE_MARCH_NATIVE)
    if(MVE_HAVE_MARCH_NATIVE)
      set(MVE_ARCH_FLAGS "-march=native")
    else()
      # Cross-compiling, or a compiler without -march=native. The campaign host
      # is an AVX2-capable Ryzen, so target that explicitly instead.
      check_cxx_compiler_flag("-mavx2 -mfma" MVE_HAVE_AVX2_FMA)
      if(MVE_HAVE_AVX2_FMA)
        set(MVE_ARCH_FLAGS "-mavx2 -mfma")
      endif()
    endif()
  endif()
  if(MVE_ARCH_FLAGS STREQUAL "")
    message(STATUS "No native architecture flag available for "
                   "'${CMAKE_SYSTEM_PROCESSOR}'; building for the default ISA")
  else()
    string(APPEND CMAKE_CXX_FLAGS " ${MVE_ARCH_FLAGS}")
  endif()
endif()

string(TOUPPER "${CMAKE_BUILD_TYPE}" MVE_BUILD_TYPE_UC)
message(STATUS "Optimization flags:${CMAKE_CXX_FLAGS} "
               "${CMAKE_CXX_FLAGS_${MVE_BUILD_TYPE_UC}}")

# External dependency: boost filesystem
find_package(Boost COMPONENTS filesystem system REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

# External dependency: Eigen3
find_package(Eigen3 REQUIRED CONFIG)

# External dependency: PCL
find_package(PCL 1.7 REQUIRED COMPONENTS common io search kdtree)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# OpenMP
find_package(OpenMP REQUIRED)

include_directories(./src)

# Evaluation executable.
add_executable(ETH3DMultiViewEvaluation
    src/accuracy.cc
    src/accuracy.h
    src/completeness.cc
    src/completeness.h
    src/fast_ply.cc
    src/fast_ply.h
    src/main.cc
    src/meshlab_project.cc
    src/meshlab_project.h
    src/tinyxml2.cpp
    src/tinyxml2.h
    src/util.h
)
target_link_libraries(ETH3DMultiViewEvaluation
    ${PCL_LIBRARIES}
    ${Boost_LIBRARIES}
    OpenMP::OpenMP_CXX
    Eigen3::Eigen
)

# ThinLTO. Checked here, after the compiler settings above are final.
if(MVE_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT MVE_IPO_OK OUTPUT MVE_IPO_MSG)
  if(MVE_IPO_OK)
    set_property(TARGET ETH3DMultiViewEvaluation
                 PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
  else()
    message(STATUS "IPO/LTO not supported, skipping: ${MVE_IPO_MSG}")
  endif()
endif()
