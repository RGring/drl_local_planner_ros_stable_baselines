#! /bin/bash
agent_id=$1

INPUT=./training_params/ppo_params.csv
OLDIFS=$IFS
IFS=,
[ ! -f $INPUT ] && { echo "$INPUT file not found"; }
init=false

while read agent_name total_timesteps policy gamma timesteps_per_actorbatch clip_param entcoeff optim_epochs optim_stepsize optim_batchsize lam adam_epsilon schedule robot_radius rew_fnc num_stacks disc_action_space normalize stage pretrained_model_path task_mode
do
    if [ "$init" = false ] ;
    then
        init=true
    fi
    if [ "$agent_name" = "$agent_id" ] ;
    then
        echo "$agent_name"
        screen -dmS ros bash -c "source ./ros.sh sim1 $policy"
        screen -S ros -X logfile screenlog_"$agent_name"_ros.log
        screen -S ros -X log
        sleep 1
        screen -dmS python bash -c "source ./ppo_training.sh $agent_name $total_timesteps $policy $gamma $timesteps_per_actorbatch $clip_param $entcoeff $optim_epochs $optim_stepsize $optim_batchsize $lam $adam_epsilon $schedule $robot_radius $rew_fnc $num_stacks $disc_action_space $normalize $stage $pretrained_model_path $task_mode"
        screen -S python -X logfile screenlog_"$agent_name"_py.log
        screen -S python -X log
        break;
    fi
done < $INPUT
bash
