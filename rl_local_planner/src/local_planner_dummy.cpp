/*********************************************************************
*
* 
* Ronja Gueldenring
*
*
*********************************************************************/
#include <pluginlib/class_list_macros.h>

#include <rl_local_planner/local_planner_dummy.h>
#include <base_local_planner/goal_functions.h>
#include <math.h>
#include <geometry_msgs/Twist.h>

//register this planner as a BaseLocalPlanner plugin
PLUGINLIB_EXPORT_CLASS(rl_local_planner::LocalPlannerDummy, nav_core::BaseLocalPlanner)
namespace rl_local_planner {

	LocalPlannerDummy::LocalPlannerDummy(): nh_(){

	}

	void LocalPlannerDummy::initialize(std::string name, tf::TransformListener* tf,
		costmap_2d::Costmap2DROS* costmap_ros){
		//costmap actually not needed.

		tf_ = tf;
		safety_stop_ = new SafetyStop(nh_, 0.6);
		goal_threshold_ = 0;
		look_ahead_distance_ = 0.5;
		path_frame_ = "";
		robot_frame_ = "base_footprint";
		ROS_INFO("LocalPlannerDummy initialized.");

	} // initialize

   	bool LocalPlannerDummy::isGoalReached(){

		tf::StampedTransform transform;
		try{
		tf_->lookupTransform(robot_frame_, path_frame_,  
								ros::Time(0), transform);
		}
		catch (tf::TransformException ex){
			ROS_ERROR("%s",ex.what());
			ros::Duration(1.0).sleep();
		}

		tf::Vector3 original_goal_transformed = transform * original_goal_;

		if(metric_dist(original_goal_transformed.getX(), original_goal_transformed.getY()) < goal_threshold_)
			return true;
		
		return false;

  	} //isGoalReached

  	bool LocalPlannerDummy::setPlan(const std::vector<geometry_msgs::PoseStamped>& orig_plan, double goal_threshold)
  	{
		goal_threshold_ = goal_threshold;
		if ( orig_plan.size() < 1 ){
			ROS_ERROR_STREAM("LocalPlannerDummy: Got an empty plan");
			return false;
		}
		global_plan_ = orig_plan;
		geometry_msgs::PoseStamped original_goal = global_plan_.back();
		original_goal_ = tf::Vector3(original_goal.pose.position.x,
							original_goal.pose.position.y,
							0.);

		path_frame_ = original_goal.header.frame_id;
		return true;
  	} //setPlan

	int LocalPlannerDummy::computeVelocityCommands(geometry_msgs::Twist& cmd_vel){
		cmd_vel.linear.x = 0.0;
		cmd_vel.linear.y = 0.0;
		cmd_vel.angular.z = 0.0;

		// if(safety_stop_->is_safe()){
			tf::Vector3 waypoint = get_next_waypoint();
			if(waypoint != NULL){
				double dist = metric_dist(waypoint.getX(), waypoint.getY());
				double angle = atan2(waypoint.getY(),waypoint.getX());
				cmd_vel.linear.x = dist;
				cmd_vel.linear.y = 0.0;
				cmd_vel.angular.z = angle;
			}
		// }
		return 1;

	}//computeVelocityCommands

	tf::Vector3 LocalPlannerDummy::get_next_waypoint(){
		// Get Transform from path_frame_ to robot_frame
		tf::StampedTransform transform;
		try{
		tf_->lookupTransform(robot_frame_, path_frame_,  
								ros::Time(0), transform);
		}
		catch (tf::TransformException ex){
			ROS_ERROR("%s",ex.what());
			ros::Duration(1.0).sleep();
		}

		// Find waypoint on global path with look_ahead_distance_
		// Discard all points that are too close.

		while(global_plan_.size() > 1){
			geometry_msgs::PoseStamped pot_waypoint = global_plan_[0];
			tf::Vector3 pot_waypoint_vec(pot_waypoint.pose.position.x,
				pot_waypoint.pose.position.y,
				0.);
			tf::Vector3 pot_waypoint_vec_transformed = transform*pot_waypoint_vec;
			
			if(metric_dist(pot_waypoint_vec_transformed.getX(), pot_waypoint_vec_transformed.getY()) > look_ahead_distance_){
				return pot_waypoint_vec_transformed;
			}
			global_plan_.erase(global_plan_.begin());
		}

		tf::Vector3 transformed_goal = transform*original_goal_;
		return transformed_goal;
	}//get_next_waypoint

	double LocalPlannerDummy::metric_dist(double x, double y){
		double dist = sqrt(pow(x, 2) + pow(y, 2));
		return dist; 
	}//metric_dist

}; // namespace rl_local_planner
