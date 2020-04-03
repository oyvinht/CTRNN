cmake_minimum_required( VERSION 3.10.0 )
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project( "ctrnn" VERSION 0.1 LANGUAGES CXX )

# CTRNN shared lib
add_library( ctrnn SHARED CTRNN.cpp )

# Test program
add_executable( ctrnn-test ctrnn-test.cpp)
target_link_libraries( ctrnn-test ctrnn)


