#! /bin/bash
agent_id=$1
num_sims=$2

INPUT=./training_params/ppo2_params.csv
OLDIFS=$IFS
IFS=,
[ ! -f $INPUT ] && { echo "$INPUT file not found"; }
init=false


while read agent_name total_timesteps policy gamma n_steps ent_coef learning_rate vf_coef max_grad_norm lam nminibatches noptepochs cliprange robot_radius rew_fnc num_stacks disc_action_space normalize stage pretrained_model_path task_mode
do
    if [ "$init" = false ] ;
    then
        init=true
    fi
    echo "$agent_name"
    if [ "$agent_name" = "$agent_id" ] ;
    then
        for ((i=1;i<=$num_sims;i++));
        do
            screen -dmS ros_sim$i bash -c "source ./ros.sh sim$i $policy"
            screen -S ros_sim$i -X logfile screenlog_"$agent_name"_ros_sim$i.log
            screen -S ros_sim$i -X log
            sleep 2
        done

        screen -dmS python bash -c "source ./ppo2_training.sh $agent_name $total_timesteps $policy $gamma $n_steps $ent_coef $learning_rate $vf_coef $max_grad_norm $lam $nminibatches $noptepochs $cliprange $robot_radius $rew_fnc $num_stacks $disc_action_space $normalize $stage $pretrained_model_path $task_mode $num_sims"
        screen -S python -X logfile screenlog_"$agent_name"_py.log
        screen -S python -X log
        break;
    fi
done < $INPUT
bash
