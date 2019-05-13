# What is this repository for?
* Setup to train a local planner with reinforcement learning approaches from [stable baselines](https://github.com/hill-a/stable-baselines) integrated ROS
* Training in a simulator fusion of [Flatland](https://github.com/avidbots/flatland) and [pedsim_ros](https://github.com/srl-freiburg/pedsim_ros)
* local planner has been trained on static and dynamic obstacles: [video](https://www.youtube.com/watch?v=laGrLaMaeT4)

# Required Software

* standart ROS setup (Code has been tested with ROS-kinetic on Ubuntu 16.04)
```
sudo apt install screen
```
```
sudo apt install virtualenv
```



# Setup Learning Infrstructure

1. Setup repository: 
    * Clone this repository in your src-folder of your catkin workspace
    ```
    cd <path_to_catkin_ws>/src/drl_local_planner_ros_stable_baselines
    cp .rosinstall ../
    cd ..
    rosws update
    cd <path_to_catkin_ws>
    catkin_make -DCMAKE_BUILD_TYPE=Release
    ```
    (please install required ros packages)

2. Setup virtual environment to be able to use python3 with ros
    ```
    virtualenv <path_to_venv>/venv_p3 --python=python3
    source <path_to_venv>/venv_p3/bin/activate
    <path_to_venv>/venv_p3/bin/pip install \
            pyyaml \
            rospkg \
            catkin_pkg \
            exception \
            numpy \
            tensorflow \
            gym \
            pyquaternion \ 
            cloudpickle \
            mpi4py \
            matplotlib
    cd <path_to_catkin_ws>/src/drl_local_planner_forks/stable_baselines/
    <path_to_venv>/venv_p3/bin/pip install -e .[docs,tests]
    ```
        
3. Set system-relevant variables 
    * Modify all relevant pathes rl_agent/config/path_config.ini


# Example usage

Copy the example_agents in your "path_to_checkpoint"

1. Train static agent
    * Open first terminal: 
    ```
    roscore
    ```
    * Open second terminal:
    ```
    cd <path_to_catkin_ws>/src/drl_local_planner_ros_stable_baselines/start_scripts
    ./entrypoint_ppo.sh ppo_1
    ```
    * Two screen sessions are created:
        - ROS setup with Flatland and Pedsim is started in:
            ```
            screen -r ros
            ```
        -  Python3 implementation is started in the virtual environement. 
        ```
        screen -r python
        ```
        It includes all RL-relevant as well as the communication with move_base
        - To detach from screen press: Strg+a, d.
    * Open third terminal: 
    ```
    roslaunch rl_bringup rviz.launch ns:="sim1"
    ```
    * Rviz with appropriate configuration starts and you should see the robot (grey rectangle)
    train.
    
2. Train dynamic agent with parallel multiple simulation envs
    * Open first terminal: 
    ```
    roscore
    ```
    * Open second terminal:
    ```
    cd <path_to_catkin_ws>/src/drl_local_planner_ros_stable_baselines/rl_agent/start_scripts
    ./entrypoint_ppo2.sh ppo2_1 2
    ```
    * Two screen sessions are created:
        - ROS setup with simulation 1 (Flatland and Pedsim) is started.
        ```
        screen -r ros_sim1
        ```
        - ROS setup with simulation 2 (Flatland and Pedsim) is started
        ```
        screen -r ros_sim2
        ```
        - Python3 implementation is started in virtual environements.
        ```
        screen -r python
        ```
        It includes all RL-relevant as well as the communication with move_base
        - To detach from screen press: Strg+a, d.
    * Open third terminal to see what happens in simulation 2:
    ```
     roslaunch rl_bringup rviz.launch ns:="sim2" 
    ```
     
    * Rviz with appropriate configuration starts and you should see the robot (grey rectangle)
    train.

3. Execute trained ppo1-agent with static objects
    * set train_mode: 0 in rl_bringup/config/rl_params_img_dyn.yaml
    * Open first terminal: 
    ```
    roscore
    ```
    * Open second terminal: 
    ```
    roslaunch rl_bringup setup.launch ns:="sim1" rl_params:="rl_params_img"
    ```
    * Open third terminal:
    ```
    source <path_to_venv>/bin/activate 
    roslaunch rl_agent run_ppo_agent.launch mode:="train"
    ```
    * Open fourth terminal: 
    ```
    roslaunch rl_bringup rviz.launch ns:="sim1"
    ```
    * Set 2D Navigation Goal in rviz

4. Execute trained ppo2-agent with pedestrians
    * set train_mode: 0 in rl_bringup/config/rl_params_img_dyn.yaml
    * Open first terminal: 
    ```
    roscore
    ```
    * Open second terminal: 
    ```
    roslaunch rl_bringup setup.launch ns:="sim1" rl_params:="rl_params_img_dyn"
    ```
    * Open third terminal:
    ```
    source <path_to_venv>/bin/activate 
    roslaunch rl_agent run_ppo2_agent.launch mode:="train"
    ```
    * Open fourth terminal: roslaunch rl_bringup rviz.launch ns:="sim1"
    * Set 2D Navigation Goal in rviz
    
    
