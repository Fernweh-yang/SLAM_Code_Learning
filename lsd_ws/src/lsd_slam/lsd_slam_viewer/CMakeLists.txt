cmake_minimum_required(VERSION 3.0.2)
project(lsd_slam_viewer)

add_compile_options(-std=c++17)
set(CMAKE_BUILD_TYPE Release)

ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/thirdparty/Sophus)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
set(OpenGL_GL_PREFERENCE GLVND)

find_package(catkin REQUIRED
  COMPONENTS
  cv_bridge
  dynamic_reconfigure
  sensor_msgs
  roscpp
  rosbag
  message_generation
  roslib
)

find_package(OpenGL REQUIRED)
set(QT_USE_QTOPENGL TRUE)
set(QT_USE_QTXML TRUE)
find_package(QGLViewer REQUIRED)
include_directories( "/usr/include/eigen3" )
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS thread)
find_package(Qt5 COMPONENTS Core Xml OpenGL Gui Widgets REQUIRED)

include_directories(
  ${QGLVIEWER_INCLUDE_DIR}
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${Qt5Core_INCLUDE_DIRS}
  ${Qt5Xml_INCLUDE_DIRS}
  ${Qt5OpenGL_INCLUDE_DIRS}
  ${Qt5Gui_INCLUDE_DIRS}
  ${Qt5Widgets_INCLUDE_DIRS}
)


# SSE flags
set(CMAKE_CXX_FLAGS
   "${CMAKE_CXX_FLAGS} -Wall"
)

add_message_files(
  DIRECTORY msg
  FILES
  keyframeMsg.msg
  keyframeGraphMsg.msg
)
generate_messages(DEPENDENCIES)

generate_dynamic_reconfigure_options(
  cfg/LSDSLAMViewerParams.cfg
)

catkin_package(
  DEPENDS EIGEN3 OpenCV OPENGL
)

# Sources files
set(SOURCE_FILES
  src/PointCloudViewer.cpp
  src/KeyFrameDisplay.cpp
  src/KeyFrameGraphDisplay.cpp
  src/settings.cpp
)

set(HEADER_FILES
  src/PointCloudViewer.h
  src/KeyFrameDisplay.h
  src/KeyFrameGraphDisplay.h
  src/settings.h
)

include_directories(
  ${PROJECT_SOURCE_DIR}/thirdparty/Sophus
)

add_executable(viewer src/main_viewer.cpp ${SOURCE_FILES} ${HEADER_FILES})

# Make sure messages are built first
add_dependencies(viewer ${${PROJECT_NAME}_EXPORTED_TARGETS})

target_link_libraries(viewer
  ${QGLViewer_LIBRARIES}
  ${QGLVIEWER_LIBRARY}
  ${catkin_LIBRARIES}
  ${Boost_LIBRARIES}
  ${Qt5Core_LIBRARIES}
  ${Qt5OpenGL_LIBRARIES}
  ${Qt5Gui_LIBRARIES}
  ${Qt5Widgets_LIBRARIES}
  GL glut GLU
)

#add_executable(videoStitch src/main_stitchVideos.cpp)
#target_link_libraries(viewer ${QGLViewer_LIBRARIES}
#			     ${QGLVIEWER_LIBRARY}
#			     ${catkin_LIBRARIES}
#			     ${QT_LIBRARIES}
#			     GL glut GLU
#		      )

