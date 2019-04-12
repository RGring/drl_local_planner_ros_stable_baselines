 /*
 * @name	 	  toggle_setup_init.cpp
 * @brief	 	  Simulation will be triggered for n_sec for initialize setup.
 * @author  	Ronja Gueldenring
 * @date 		  2019/04/05
 **/
#include <ros/ros.h>
#include <flatland_msgs/Step.h>

int main(int argc, char** argv){
    
  ros::init(argc, argv, "map_echo");
  ros::NodeHandle node;

  std::string train_mode_topic = ros::this_node::getNamespace() + "/rl_agent/train_mode";
  int rl_mode;
  node.getParam(train_mode_topic, rl_mode);

  bool keep_clock_running = false;
  if(rl_mode == 2){
    keep_clock_running = true;
  }
  
  float n_sec = 10.0;
  ros::ServiceClient step_simulation_ = node.serviceClient<flatland_msgs::Step>("step");
  flatland_msgs::Step msg;
  msg.request.step_time.data = 0.1;
  ros::WallTime begin = ros::WallTime::now();
  ros::WallRate r(30);
  while ((ros::WallTime::now() - begin).toSec() < n_sec || keep_clock_running) {
      if(!step_simulation_.call(msg)){
        ROS_ERROR("Failed to call step_simulation_ service");
      }
    r.sleep();
  }

  return 0;
};
