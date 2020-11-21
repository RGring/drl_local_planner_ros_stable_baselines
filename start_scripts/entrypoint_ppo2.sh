#! /bin/bash
#get config-params
path_to_venv=$(awk -F "=" '/path_to_venv/ {print $2}' ../rl_bringup/config/path_config.ini)
path_to_catkin_ws=$(awk -F "=" '/path_to_catkin_ws/ {print $2}' ../rl_bringup/config/path_config.ini)
ros_ver=$(awk -F "=" '/ros_version/ {print $2}' ../rl_bringup/config/path_config.ini)

# source ros stuff
source /opt/ros/$ros_ver/setup.bash
source $path_to_catkin_ws/devel/setup.bash
source $path_to_venv/bin/activate

agent_id=$1
num_sims=$2

INPUT=./training_params/ppo2_params.csv
OLDIFS=$IFS
IFS=,
[ ! -f $INPUT ] && { echo "$INPUT file not found"; }
init=false

# Checking if roscore is running. If not roscore is started
if ! rostopic list ; 
then
    screen -dmS roscore bash -c "source ./start_roscore.sh"
    screen -S roscore -X logfile screenlog_roscore.log
    screen -S roscore -X log
    sleep 4
fi

# Loading training maps
MAPS=./training_params/training_maps.csv
maps=()
while read map
do
    if [ -z "$map" ];
    then
        break
    fi
    maps+=($map)
done < $MAPS

while read agent_name total_timesteps policy gamma n_steps ent_coef learning_rate vf_coef max_grad_norm lam nminibatches noptepochs cliprange robot_radius rew_fnc num_stacks stack_offset disc_action_space normalize stage pretrained_model_path task_mode
do
    if [ "$init" = false ] ;
    then
        init=true
        continue;
    fi

    if [ "$agent_name" = "$agent_id" ] || [ "$agent_id" = "all" ];
    then
        echo "$agent_name"
        echo "$policy"

        # Starting num_sims simulations with different maps.
        i_map=0
        for ((i=1;i<=$num_sims;i++));
        do
            echo "${maps[i_map]}"
            screen -dmS ros_sim$i bash -c "source ./ros.sh sim$i $policy ${maps[i_map]}"
            screen -S ros_sim$i -X logfile screenlog_"$agent_name"_ros_sim$i.log
            screen -S ros_sim$i -X log
            i_map=$((i_map+1))
            if [ "$i_map" -eq "${#maps[@]}" ] ;
            then
                i_map=0
            fi
            sleep 5
        done

        # Starting Reinforcement Learning training (PPO2)
        screen -dmS python bash -c "source ./ppo2_training.sh $agent_name $total_timesteps $policy $gamma $n_steps $ent_coef $learning_rate $vf_coef $max_grad_norm $lam $nminibatches $noptepochs $cliprange $robot_radius $rew_fnc $num_stacks $stack_offset $disc_action_space $normalize $stage $pretrained_model_path $task_mode $num_sims"
        screen -S python -X logfile screenlog_"$agent_name"_py.log
        screen -S python -X log

        # Wait until training is done
        while (screen -list | grep -q "python");
        do
        echo "sleep"
        sleep 5
        done

        # Old training setup is closed, so that new one can be started.
        for ((i=1;i<=$num_sims;i++));
        do
            screen -X -S ros_sim$i quit
        done
    fi
done < $INPUT
#bash
