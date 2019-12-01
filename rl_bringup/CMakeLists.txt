cmake_minimum_required(VERSION 2.8.3)
project(rl_bringup)

set(PACKAGE_DEPS
  roscpp
  rospy
  std_msgs
  nav_msgs
  tf
  flatland_msgs
)

find_package(catkin REQUIRED COMPONENTS ${PACKAGE_DEPS})

catkin_package(
  CATKIN_DEPENDS ${PACKAGE_DEPS}
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config/
)

###########
## Build ##
###########

include_directories(${catkin_INCLUDE_DIRS})
link_directories(${catkin_LIBRARY_DIRS})

set(TOGGLE_SETUP_INIT toggle_setup_init)
add_executable(${TOGGLE_SETUP_INIT} src/toggle_setup_init.cpp)
add_dependencies(${TOGGLE_SETUP_INIT} ${catkin_EXPORTED_TARGETS})
target_link_libraries(${TOGGLE_SETUP_INIT} ${catkin_LIBRARIES})
