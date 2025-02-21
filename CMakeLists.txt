cmake_minimum_required(VERSION 3.10)
project(AMD_MatMul_Comparison LANGUAGES CXX)

# Find HIP package
find_package(HIP REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Add executables
add_executable(cpu_matmul src/cpu/cpu_matmul.cpp)
add_executable(naive_matmul src/naive/naive_matmul.cpp)
add_executable(tiled_matmul src/tiled/tiled_matmul.cpp)
add_executable(optimized_matmul src/optimized/optimized_matmul.cpp)

# Link HIP to GPU implementations
target_link_libraries(naive_matmul hip::device)
target_link_libraries(tiled_matmul hip::device)
target_link_libraries(optimized_matmul hip::device)

# Set output directory for binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin) 