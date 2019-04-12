/*
 * @name	 	wp_generator.cpp
 * @brief	 	Generation of waypoints on the global path and 
 * 				determination of the closest waypoints to the robot.
 * @author 		Ronja Gueldenring
 * @date 		2019/04/05
 **/
#include <rl_local_planner/wp_generator.h>


namespace rl_local_planner {

	WpGenerator::WpGenerator(const ros::NodeHandle& node_handle)
							: nh_{node_handle}{
        
		// Get params of parameter server
		nh_.getParam("rl_agent/look_ahead_distance", look_ahead_distance_);
		nh_.getParam("rl_agent/wp_reached_dist", wp_reached_dist_);
		nh_.getParam("rl_agent/num_of_wps", num_of_wps_);
		nh_.getParam("rl_agent/robot_frame", robot_frame_);

		// Services
		std::string global_plan_service_name = ros::this_node::getName() + "/set_gobal_plan";
		set_path_service = nh_.advertiseService(global_plan_service_name, &WpGenerator::path_callback_, this);
        
		//Publisher
		wp_pub_ = nh_.advertise<rl_msgs::Waypoint>("wp", 1, true);
		wp_reached_pub_ = nh_.advertise<rl_msgs::Waypoint>("wp_reached", 1, true);
	}

	bool WpGenerator::path_callback_(rl_msgs::SetPath::Request& request, rl_msgs::SetPath::Response& response){
		if(request.path.poses.size() == 0){
				return false;
		}
		path_frame_ = request.path.header.frame_id;
        geometry_msgs::PoseStamped goal_temp = request.path.poses.back();
		goal_ = tf::Vector3(goal_temp.pose.position.x,
							goal_temp.pose.position.y,
							0.);
        precalculate_waypoints_(request.path.poses);
        get_closest_wps();
        return true;
    }


	void WpGenerator::precalculate_waypoints_(std::vector<geometry_msgs::PoseStamped> global_plan){
		waypoints_.clear();
		geometry_msgs::PoseStamped prev = global_plan[0];
		double dist = 0.0;
		for(int i = 1; i < global_plan.size(); i++){
			dist += metric_dist((prev.pose.position.x - global_plan[i].pose.position.x),
							(prev.pose.position.y - global_plan[i].pose.position.y));
			if(dist > look_ahead_distance_){
				dist = 0.0;
				waypoints_.push_back(global_plan[i]);
			}
			prev = global_plan[i];
		}
		for(int i = 0; i < num_of_wps_ + 1; i++){
        	waypoints_.push_back(global_plan.back());
		}
		return;
	}

	void WpGenerator::get_closest_wps(){
		if(waypoints_.size() < num_of_wps_){
            return;
        }

		//Prepare Waypoint-message that will be published
		rl_msgs::Waypoint msg;
		msg.header.frame_id = robot_frame_;
		msg.header.stamp = ros::Time::now();
        msg.is_new.data = false;

		// Get Transform from path_frame_ to robot_frame
		tf::StampedTransform transform;
		try{
		    tf_.lookupTransform(robot_frame_, path_frame_,  
								ros::Time(0), transform);

		}
		catch (tf::TransformException ex){
			ROS_ERROR("%s",ex.what());
			ros::WallDuration(1.0).sleep();
		}

		//Find closest point on path
		tf::Vector3 closest_wp = get_transformed_wp_(waypoints_[0], transform);
		double dist = metric_dist(closest_wp.getX(), closest_wp.getY());
		for(int i = 1; i < waypoints_.size(); i++){
			tf::Vector3 temp_wp = get_transformed_wp_(waypoints_[i], transform);
			double temp_dist = metric_dist(temp_wp.getX(), temp_wp.getY());
			if (temp_dist < dist){
				dist = temp_dist;
				closest_wp = temp_wp;
			}else{
				//Closest point on path found!
				int start_index = i - 1;

				//Is closest wp reached in a certain radius?
				if (dist < wp_reached_dist_){
					msg.is_new.data = true;
					for (int k = 0; k < i; k++){
						waypoints_.erase(waypoints_.begin());
					}
					start_index = 0;	
				}

				// Is closest waypoint already overtaken? Then take next one.
				tf::Vector3 second_wp = get_transformed_wp_(waypoints_[start_index + 1], transform);
				double dist_second_wp = metric_dist(temp_wp.getX(), temp_wp.getY());
				if (dist + look_ahead_distance_ > dist_second_wp){
					start_index +=1;
				}

				//Getting all sequencing waypoints.
				std::vector<geometry_msgs::Point> points;
				for(int j = start_index; j < start_index + num_of_wps_; j++){
					geometry_msgs::Point p;
					if (j < waypoints_.size()){
						tf::Vector3 wp = get_transformed_wp_(waypoints_[j], transform);
						p.x = wp.getX();
						p.y = wp.getY();
						points.push_back(p);
					}
				}
				msg.points = points;
				if (msg.is_new.data == true){
					wp_reached_pub_.publish(msg);
				}else{
					wp_pub_.publish(msg);
				}
				
				break;
			}
		}
		return;		
	}

	tf::Vector3 WpGenerator::get_transformed_wp_(geometry_msgs::PoseStamped wp, tf::StampedTransform transform){
		tf::Vector3 wp_vec(wp.pose.position.x,
				wp.pose.position.y,
				0.);
		tf::Vector3 wp_vec_transformed = transform*wp_vec;
		return wp_vec_transformed;
	}

    double WpGenerator::metric_dist(double x, double y){
		double dist = sqrt(pow(x , 2) + pow(y , 2));
		return dist; 
	}//metric_dist

}; // namespace rl_local_planner

int main(int argc, char** argv){
  ros::init(argc, argv, "wp_generator");
  ros::NodeHandle node;
  int rl_mode;
  std::string train_mode_topic = ros::this_node::getNamespace() + "/rl_agent/train_mode";
  node.param(train_mode_topic, rl_mode, 1);
  rl_local_planner::WpGenerator wg(node);
	ros::WallRate r(100);
	while (ros::ok()) {
			wg.get_closest_wps();
			ros::spinOnce();
			r.sleep();
	}
	return 0;
  
};