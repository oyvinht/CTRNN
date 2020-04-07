cmake_minimum_required( VERSION 3.10.0 )
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project( "ctrnn" VERSION 0.1 LANGUAGES CXX CUDA )

# CTRNN shared lib
add_library( ctrnn SHARED CTRNN.cu )
set_target_properties( ctrnn PROPERTIES CUDA_SEPARABLE_COMPILATION ON )

# Test program
add_executable( ctrnn-test ctrnn-test.cpp)
set_property(TARGET ctrnn-test PROPERTY CUDA_SEPARABLE_COMPILATION ON )
target_link_libraries( ctrnn-test ctrnn)


