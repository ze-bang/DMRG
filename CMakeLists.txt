cmake_minimum_required(VERSION 3.10)
project(DMRG)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the ITensor library
find_package(ITensor REQUIRED)

# Add the executable
add_executable(dmrg main.cpp)

# Add include directories
target_include_directories(dmrg PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Link ITensor library to the executable
target_link_libraries(dmrg ITensor::itensor)
