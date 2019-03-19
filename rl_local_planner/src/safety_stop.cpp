/*********************************************************************
*
* 
* Ronja Gueldenring
*
*
*********************************************************************/

#include <rl_local_planner/safety_stop.h>
#include <iostream>
#include <geometry_msgs/Point32.h>


namespace rl_local_planner {

	SafetyStop::SafetyStop(const ros::NodeHandle& node_handle, double thresh)
							: nh_{node_handle}, thresh_{thresh}{
		safe_ = false;
		f_safe_ = false;
		b_safe_ = false;
		sub_f_scan = nh_.subscribe("f_scan", 1, &SafetyStop::f_scan_callback, this);
      	sub_b_scan = nh_.subscribe("b_scan", 1, &SafetyStop::b_scan_callback, this);
	}

	bool SafetyStop::is_safe(){
		return safe_;
	}

	double SafetyStop::get_thesh(){
		return thresh_;
	}

	void SafetyStop::f_scan_callback(const sensor_msgs::LaserScanConstPtr& laser){
		sensor_msgs::PointCloud cloud = laserToCloud(laser, "base_footprint");
		f_safe_ = is_scan_safe(cloud.points);
		set_safe();

	}
	void SafetyStop::b_scan_callback(const sensor_msgs::LaserScanConstPtr& laser){
		const sensor_msgs::PointCloud cloud = laserToCloud(laser, "base_footprint");
		b_safe_ = is_scan_safe(cloud.points);
		set_safe();

	}

	sensor_msgs::PointCloud SafetyStop::laserToCloud(const sensor_msgs::LaserScanConstPtr& laser, std::string target_frame){
		sensor_msgs::PointCloud cloud;

		if(!listener_.waitForTransform(laser->header.frame_id, target_frame,
			laser->header.stamp + ros::Duration().fromSec(laser->ranges.size()*laser->time_increment),
			ros::Duration(1.0))){
			ROS_WARN("waitForTransform in SafetyStop::laserToCloud failed.");
			return cloud;
		}

		
  		projector_.transformLaserScanToPointCloud(target_frame, *laser, cloud, listener_);
		return cloud;
	}

	bool SafetyStop::is_scan_safe(const std::vector<geometry_msgs::Point32> vec){
		for(int i = 0; i < vec.size(); i++){
			double dist = sqrt(pow(vec[i].x, 2) + pow(vec[i].y, 2));
			if (dist < thresh_){
				return false;
			}
		}
		return true;
	}

	void SafetyStop::set_safe(){
		if(b_safe_ && f_safe_){
			safe_ = true;
		}else{
			safe_ = false;
		}
	}
	     
	      

}; // namespace rl_local_planner
