#! /bin/sh

#get config-params
path_to_catkin_ws=$(awk -F "=" '/path_to_catkin_ws/ {print $2}' ../rl_bringup/config/path_config.ini)
ros_ver=$(awk -F "=" '/ros_version/ {print $2}' ../rl_bringup/config/path_config.ini)

# source ros stuff
source /opt/ros/$ros_ver/setup.bash
source $path_to_catkin_ws/devel/setup.bash

roscore