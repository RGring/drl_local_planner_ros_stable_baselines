cmake_minimum_required(VERSION 2.8.3)
project(rl_agent)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

# Get the information about this package's buildtime dependencies
  find_package(catkin REQUIRED
    COMPONENTS 
    rospy 
    roscpp 
    message_generation 
    std_msgs 
    sensor_msgs 
    laser_geometry
    geometry_msgs
    nav_msgs
    rl_msgs
    tf
    pcl_ros
  )

  find_package(PCL REQUIRED)

  catkin_python_setup()


  # Declare that this catkin package's runtime dependencies
  catkin_package(
   CATKIN_DEPENDS 
   stable_baselines 
   message_runtime 
   std_msgs 
   rl_msgs 
   sensor_msgs 
   geometry_msgs
  )

  include_directories(
    ${catkin_INCLUDE_DIRS}
    include
    scripts
    )

  link_directories(${catkin_LIBRARY_DIRS})

  add_executable(laser_scan_merger
    src/laser_scan_merger.cpp
  )

  add_dependencies(laser_scan_merger
    ${${PROJECT_NAME}_EXPORTED_TARGETS}
    ${catkin_EXPORTED_TARGETS}
  )

  target_link_libraries(laser_scan_merger
    ${catkin_LIBRARIES}
  )

  add_executable(tf_python
    src/tf_python.cpp
  )

  add_dependencies(tf_python
    ${${PROJECT_NAME}_EXPORTED_TARGETS}
    ${catkin_EXPORTED_TARGETS}
  )

  target_link_libraries(tf_python
    ${catkin_LIBRARIES}
  )

  catkin_install_python(PROGRAMS
    scripts/train_scripts/train_ppo.py
    scripts/run_scripts/run_ppo.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
  
  install(DIRECTORY launch/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch/
  )

  install(PROGRAMS
    start_scripts/run_launch_in_venv.sh
    start_scripts/run_launch_in_venv_robot.sh
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/start_scripts/
  )

  install(
  TARGETS
    laser_scan_merger
    tf_python
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)