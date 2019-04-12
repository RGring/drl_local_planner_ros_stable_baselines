cmake_minimum_required(VERSION 2.8.3)
project(rl_msgs)

# Ensure we're using c++11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  std_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  message_generation
)

## System dependencies are found with CMake's conventions
find_package(PkgConfig REQUIRED)

################
## Build MSGS ##
################
add_message_files(FILES
  Waypoint.msg
)

add_service_files(FILES
  MergeScans.srv
  SetPath.srv
  StateImageGenerationSrv.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  nav_msgs 
  sensor_msgs
  rl_msgs
)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS
  CATKIN_DEPENDS 
  std_msgs 
  geometry_msgs 
  nav_msgs 
  sensor_msgs 
  message_runtime
)

###########
## Build ##
###########

include_directories(
  ${catkin_INCLUDE_DIRS}
)
