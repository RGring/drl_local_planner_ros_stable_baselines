#! /bin/sh
sim_name=$1
policy=$2
map=$3

#get config-params
path_to_catkin_ws=$(awk -F "=" '/path_to_catkin_ws/ {print $2}' ../rl_bringup/config/path_config.ini)
ros_ver=$(awk -F "=" '/ros_version/ {print $2}' ../rl_bringup/config/path_config.ini)

# source ros stuff
source /opt/ros/$ros_ver/setup.bash
source $path_to_catkin_ws/devel/setup.bash

map_path=$path_to_catkin_ws/src/drl_local_planner_ros_stable_baselines/flatland_setup/maps/$map
echo "map_path"
echo "$map_path"
echo "$sim_name"

if [ "$policy" = "CnnPolicy_multi_input_vel" ] || [ "$policy" = "CnnPolicy_multi_input_vel2" ] || [ "$policy" = "CnnPolicy" ];
then
    echo "$policy"
    roslaunch rl_bringup setup.launch ns:="$sim_name" rl_params:="rl_params_img" map_path:="$map_path"
fi

if [ "$policy" = "CNN1DPolicy" ] || [ "$policy" = "CNN1DPolicy_multi_input" ];
then
    echo "$policy"
    roslaunch rl_bringup setup.launch ns:="$sim_name" rl_params:="rl_params_scan" map_path:="$map_path"
fi

