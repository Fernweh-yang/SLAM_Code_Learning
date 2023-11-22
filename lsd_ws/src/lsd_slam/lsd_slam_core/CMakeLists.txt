cmake_minimum_required(VERSION 3.0.2)
project(lsd_slam_core)

# debug
set(CMAKE_BUILD_TYPE "Debug")
add_compile_options(-std=gnu++17) # fixes some errors about allocator types appearing in newer g++ versions
#add_compile_options(-march=native)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  dynamic_reconfigure
  sensor_msgs
  image_transport
  roscpp
  rosbag
  lsd_slam_viewer
)
include_directories( "/usr/include/eigen3" )
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(X11 REQUIRED)
include(cmake/FindG2O.cmake)
include(cmake/FindSuiteParse.cmake)

message(STATUS "G2O_INCLUDE_DIR : " ${G2O_INCLUDE_DIR})
message(STATUS "OpenCV_INCLUDE_DIRS : " ${OpenCV_INCLUDE_DIRS})
message(STATUS "CHOLMOD_INCLUDE_DIR : " ${CHOLMOD_INCLUDE_DIR})
message(STATUS "CSPARSE_INCLUDE_DIR : " ${CSPARSE_INCLUDE_DIR})

# Sophus
add_subdirectory(${PROJECT_SOURCE_DIR}/thirdparty/Sophus)
include_directories(${PROJECT_SOURCE_DIR}/thirdparty/Sophus/sophus)

# FabMap
# uncomment this part to enable fabmap
#add_subdirectory(${PROJECT_SOURCE_DIR}/thirdparty/openFabMap)
#include_directories(${PROJECT_SOURCE_DIR}/thirdparty/openFabMap/include)
#add_definitions("-DHAVE_FABMAP")
#set(FABMAP_LIB openFABMAP )

generate_dynamic_reconfigure_options(
  cfg/LSDDebugParams.cfg
  cfg/LSDParams.cfg
)

catkin_package(
  LIBRARIES lsdslam
  DEPENDS EIGEN3 CHOLMOD
  CATKIN_DEPENDS libg2o
)

# SSE flags
add_definitions("-DUSE_ROS")
add_definitions("-DENABLE_SSE")

# Set source files
set(lsd_SOURCE_FILES
  ${PROJECT_SOURCE_DIR}/src/DataStructures/Frame.cpp
  ${PROJECT_SOURCE_DIR}/src/DataStructures/FramePoseStruct.cpp
  ${PROJECT_SOURCE_DIR}/src/DataStructures/FrameMemory.cpp
  ${PROJECT_SOURCE_DIR}/src/SlamSystem.cpp
  ${PROJECT_SOURCE_DIR}/src/LiveSLAMWrapper.cpp
  ${PROJECT_SOURCE_DIR}/src/DepthEstimation/DepthMap.cpp
  ${PROJECT_SOURCE_DIR}/src/DepthEstimation/DepthMapPixelHypothesis.cpp
  ${PROJECT_SOURCE_DIR}/src/util/globalFuncs.cpp
  ${PROJECT_SOURCE_DIR}/src/util/SophusUtil.cpp
  ${PROJECT_SOURCE_DIR}/src/util/settings.cpp
  ${PROJECT_SOURCE_DIR}/src/util/Undistorter.cpp
  ${PROJECT_SOURCE_DIR}/src/Tracking/Sim3Tracker.cpp
  ${PROJECT_SOURCE_DIR}/src/Tracking/Relocalizer.cpp
  ${PROJECT_SOURCE_DIR}/src/Tracking/SE3Tracker.cpp
  ${PROJECT_SOURCE_DIR}/src/Tracking/least_squares.cpp
  ${PROJECT_SOURCE_DIR}/src/Tracking/TrackingReference.cpp
  ${PROJECT_SOURCE_DIR}/src/IOWrapper/Timestamp.cpp
  ${PROJECT_SOURCE_DIR}/src/GlobalMapping/FabMap.cpp
  ${PROJECT_SOURCE_DIR}/src/GlobalMapping/KeyFrameGraph.cpp
  ${PROJECT_SOURCE_DIR}/src/GlobalMapping/g2oTypeSim3Sophus.cpp
  ${PROJECT_SOURCE_DIR}/src/GlobalMapping/TrackableKeyFrameSearch.cpp
)
set(SOURCE_FILES
  ${lsd_SOURCE_FILES}
  ${PROJECT_SOURCE_DIR}/src/IOWrapper/ROS/ROSImageStreamThread.cpp
  ${PROJECT_SOURCE_DIR}/src/IOWrapper/ROS/ROSOutput3DWrapper.cpp
  ${PROJECT_SOURCE_DIR}/src/IOWrapper/OpenCV/ImageDisplay_OpenCV.cpp
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${G2O_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${CSPARSE_INCLUDE_DIR} #Has been set by SuiteParse
  ${CHOLMOD_INCLUDE_DIR} #Has been set by SuiteParse
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/src/DataStructures
  ${PROJECT_SOURCE_DIR}/src/DepthEstimation
  ${PROJECT_SOURCE_DIR}/src/GlobalMapping
  ${PROJECT_SOURCE_DIR}/src/IOWrapper
  ${PROJECT_SOURCE_DIR}/src/Tracking
  ${PROJECT_SOURCE_DIR}/src/util
  ${PROJECT_SOURCE_DIR}/thirdparty/Sophus
  ${PROJECT_SOURCE_DIR}/thirdparty/Sophus/sophus
)

# build shared library.
add_library(lsdslam SHARED ${SOURCE_FILES})
target_link_libraries(lsdslam
  ${FABMAP_LIB}
  ${G2O_LIBRARIES}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${CHOLMOD_LIBRARY}
  ${CSPARSE_LIBRARY}
)

# build live ros node
add_executable(live_slam src/main_live_odometry.cpp)
target_link_libraries(live_slam lsdslam ${catkin_LIBRARIES} ${G2O_LIBRARIES} ${X11_LIBRARIES})

# build image node
add_executable(dataset src/main_on_images.cpp)
target_link_libraries(dataset lsdslam ${catkin_LIBRARIES} ${G2O_LIBRARIES})


