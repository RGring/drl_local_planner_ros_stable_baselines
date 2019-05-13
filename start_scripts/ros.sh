#! /bin/sh
sim_name=$1
policy=$2

path_to_catkin_ws=$(awk -F "=" '/path_to_catkin_ws/ {print $2}' ../rl_agent/config/path_config.ini)
ros_ver=$(awk -F "=" '/ros_version/ {print $2}' ../rl_agent/config/path_config.ini)
# source ros stuff
source /opt/ros/$ros_ver/setup.bash
source $path_to_catkin_ws/devel/setup.bash

echo "$sim_name"

if [ "$policy" = "CnnPolicy" ] ;
then
    echo "$policy"
    roslaunch rl_bringup setup.launch ns:="$sim_name" rl_params:="rl_params_img"
fi

if [ "$policy" = "CnnPolicy_multi_input_vel" ] ;
then
    echo "$policy"
    roslaunch rl_bringup setup.launch ns:="$sim_name" rl_params:="rl_params_img_dyn"
fi

if [ "$policy" = "CnnPolicy_multi_input_vel2" ] ;
then
    echo "$policy"
    roslaunch rl_bringup setup.launch ns:="$sim_name" rl_params:="rl_params_img_dyn"
fi

if [ "$policy" = "CnnLstmPolicy" ] ;
then
    echo "$policy"
    roslaunch rl_bringup setup.launch ns:="$sim_name" rl_params:="rl_params_img"
fi

if [ "$policy" = "CNN1DPolicy_multi_input" ] ;
then
    echo "$policy"
    roslaunch rl_bringup setup.launch ns:="$sim_name" rl_params:="rl_params_scan"
fi

if [ "$policy" = "CNN1DPolicy" ] ;
then
    echo "$policy"
    roslaunch rl_bringup setup.launch ns:="$sim_name" rl_params:="rl_params_scan"
fi

