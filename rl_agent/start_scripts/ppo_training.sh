#! /bin/sh

# source ros stuff
source /opt/ros/kinetic/setup.bash
source ~/catkin_ws/devel/setup.bash
source ~/venv_p3/bin/activate

# get params
agent_name=$1
total_timesteps=$2
policy=$3
gamma=$4
timesteps_per_actorbatch=$5
clip_param=$6
entcoeff=$7
optim_epochs=$8
optim_stepsize=$9
optim_batchsize=${10}
lam=${11}
adam_epsilon=${12}
schedule=${13}
robot_radius=${14}
rew_fnc=${15}
num_stacks=${16}
disc_action_space=${17}
normalize=${18}
stage=${19}
pretrained_model_path=${20}
task_mode=${21}

# start rl_agent
python ../scripts/train_scripts/train_ppo1.py $agent_name $total_timesteps $policy $gamma $timesteps_per_actorbatch $clip_param $entcoeff $optim_epochs $optim_stepsize $optim_batchsize $lam $adam_epsilon $schedule $robot_radius $rew_fnc $num_stacks $disc_action_space $normalize $stage $pretrained_model_path $task_mode

deactivate
IFS=$OLDIFS


