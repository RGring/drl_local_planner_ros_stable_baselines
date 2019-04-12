#! /bin/sh
sim_name=$1
policy=$2

source /opt/ros/kinetic/setup.bash
source ~/catkin_ws/devel/setup.bash

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

