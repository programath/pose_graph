cmake_minimum_required( VERSION 2.8 )
project( pose_graph )

set( CMAKE_BUILD_TYPE Release )
#set( CMAKE_CXX_FLAGS "-std=c++11 -g -O3" )
set( CMAKE_CXX_STANDARD 14 )

include_directories( "/usr/include/eigen3" )

find_package(OpenCV 4.3.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Ceres)
include_directories(${CERES_INCLUDE_DIRS})

add_executable(gnss_odom_optimizer gnss_odom_optimizer.cc pose_local_parameterization.cc io.cc hdmap_loader.cc)
target_link_libraries(gnss_odom_optimizer ${CERES_LIBRARIES} ${OpenCV_LIBS})

#add_executable(test_loader test_loader.cc hdmap_loader.cc)
#target_link_libraries(test_loader ${OpenCV_LIBS})
