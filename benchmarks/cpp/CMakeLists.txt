cmake_minimum_required(VERSION 3.10)
project(iomm_benchmarks LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

file(GLOB BENCH_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_LIST_DIR}/src/*_bench.cpp")

if (BENCH_SOURCES)
  message(STATUS "Found benchmarks: ${BENCH_SOURCES}")
else()
  message(FATAL_ERROR "No .cpp benchmark files found in ${CMAKE_CURRENT_LIST_DIR}")
endif()

foreach (bench_src IN LISTS BENCH_SOURCES)
  get_filename_component(bench_name ${bench_src} NAME_WE)

  add_executable(${bench_name} ${bench_src})

  set_target_properties(${bench_name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )
endforeach()
