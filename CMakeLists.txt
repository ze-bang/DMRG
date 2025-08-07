cmake_minimum_required(VERSION 3.10)
project(DMRG)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define the path to the ITensor directory
set(ITENSOR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/itensor)

# Add the executable
add_executable(dmrg src/main.cpp)

# Add include directories for your project and for ITensor
target_include_directories(dmrg PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include
    ${ITENSOR_DIR}
)

# Link ITensor library and its dependencies to the executable
target_link_libraries(dmrg 
    ${ITENSOR_DIR}/lib/libitensor.a
    -llapack
    -lblas
)