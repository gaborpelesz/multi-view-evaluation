cmake_minimum_required(VERSION 3.16)
project(ETH3DMultiViewEvaluation CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Settings. RelWithDebInfo (-O2 -g) is the canonical benchmark configuration:
# it matches what benchmark/eth3d/multi-view-evaluation-compat.patch produces
# and keeps frame pointers/line tables for the profiler.
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type not specified, using RelWithDebInfo")
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

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
