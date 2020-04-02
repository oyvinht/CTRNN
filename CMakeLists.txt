cmake_minimum_required( VERSION 3.10.0 )

project( "ctrnn" VERSION 0.1 LANGUAGES CXX )

set( SHARED_LIBRARY_NAME "${PROJECT_NAME}-shared" )
add_library( lib CTRNN.cpp )

add_executable( ctrnn-test ctrnn-test.cpp CTRNN.h CTRNN.cpp)

#add_library( ${SHARED_LIBRARY_NAME} STATIC ${MANIFEST} )


