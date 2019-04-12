/*
 * @name	 	laser_scan_merger.cpp
 * @brief	 	Merges several scans to a single one with coordinate system in robot_frame
 * @author  	Ronja Gueldenring
 * @date 	  	2019/04/05
 * @comment 	Code snippets are taken from: https://github.com/iralabdisco/ira_laser_tools
 **/
#include <rl_agent/laser_scan_merger.h>

namespace rl_agent {

	LaserScanMerger::LaserScanMerger(const ros::NodeHandle& node_handle): nh_(node_handle){
		std::string merge_service_name_ = ros::this_node::getName() + "/merge_scans";
        merge_service_ = nh_.advertiseService("merge_scans", &LaserScanMerger::merge_scan_callback, this);
        nh_.getParam("rl_agent/robot_frame", robot_frame_);
		min_height_ = 0.0;
		max_height_ = 0.25;
	}

    bool LaserScanMerger::merge_scan_callback(rl_msgs::MergeScans::Request& request, 
                                    rl_msgs::MergeScans::Response& response){

        int num_laser_scan = request.scans.size();
        if (num_laser_scan <= 0){
            return false;
        }

		//Scan properties of first scan are saved for merged scan
        this->angle_increment_ = request.scans[0].angle_increment;
	    this->time_increment_ = request.scans[0].time_increment;
        this->scan_time_ = request.scans[0].scan_time;
	    this->range_min_ = request.scans[0].range_min;
	    this->range_max_ = request.scans[0].range_max;

		// Each scan is transformed to a pointcloud
        pcl::PCLPointCloud2 clouds[num_laser_scan];
        for (int i = 0; i < num_laser_scan; i++){
            sensor_msgs::LaserScan scan_in = request.scans[i];
            if(!listener_.waitForTransform(scan_in.header.frame_id, robot_frame_,
            scan_in.header.stamp + ros::Duration().fromSec(scan_in.ranges.size()*scan_in.time_increment),
            ros::Duration(1.0))){
                ROS_WARN("Duration exceeded");
                return false;
            }

            sensor_msgs::PointCloud cloud_temp_1;
            sensor_msgs::PointCloud2 cloud_temp_2;
            projector_.transformLaserScanToPointCloud(robot_frame_ , scan_in , cloud_temp_1 , listener_);
            sensor_msgs::convertPointCloudToPointCloud2(cloud_temp_1 , cloud_temp_2);
            pcl_conversions::toPCL(cloud_temp_2, clouds[i]);
        }

		// All pointclouds are concatenated
        pcl::PCLPointCloud2 merged_cloud = clouds[0];
        for(int i=1; i<num_laser_scan; ++i)
		{
			pcl::concatenatePointCloud(merged_cloud, clouds[i], merged_cloud);
		}

		// From the concatenated pointcloud, a laserscan is generated.
        Eigen::MatrixXf points;
        getPointCloudAsEigen(merged_cloud , points);
		sensor_msgs::LaserScanPtr merged_scan = pointcloud_to_laserscan(points, &merged_cloud);
        response.merged_scan = *merged_scan;
        return true;
    }//merge_scan_callback

    sensor_msgs::LaserScanPtr LaserScanMerger::pointcloud_to_laserscan(Eigen::MatrixXf points, pcl::PCLPointCloud2 *merged_cloud){
		sensor_msgs::LaserScanPtr output(new sensor_msgs::LaserScan());
		output->header = pcl_conversions::fromPCL(merged_cloud->header);
		output->header.frame_id = robot_frame_;
		output->angle_min = -M_PI;
		output->angle_max = M_PI;
		output->angle_increment = this->angle_increment_;
		output->time_increment = this->time_increment_;
		output->scan_time = this->scan_time_;
		output->range_min = this->range_min_;
		output->range_max = this->range_max_;

		uint32_t ranges_size = std::ceil((output->angle_max - output->angle_min) / output->angle_increment);
		output->ranges.assign(ranges_size, output->range_max + 1.0);

		for(int i=0; i<points.cols(); i++)
		{
			const float &x = points(0,i);
			const float &y = points(1,i);
			const float &z = points(2,i);

			if ( std::isnan(x) || std::isnan(y) || std::isnan(z) )
			{
				ROS_DEBUG("rejected for nan in point(%f, %f, %f)\n", x, y, z);
				continue;
			}

			double range_sq = y*y+x*x;
			double range_min_sq_ = output->range_min * output->range_min;
			if (range_sq < range_min_sq_) {
				ROS_DEBUG("rejected for range %f below minimum value %f. Point: (%f, %f, %f)", range_sq, range_min_sq_, x, y, z);
				continue;
			}

			double angle = atan2(y, x);
			if (angle < output->angle_min || angle > output->angle_max)
			{
				ROS_DEBUG("rejected for angle %f not in range (%f, %f)\n", angle, output->angle_min, output->angle_max);
				continue;
			}
			int index = (angle - output->angle_min) / output->angle_increment;


			if (output->ranges[index] * output->ranges[index] > range_sq)
				output->ranges[index] = sqrt(range_sq);
		}
		return output;
	}//pointcloud_to_laserscan
}; // namespace rl_agent

int main(int argc, char** argv){
  ros::init(argc, argv, "laser_scan_merger");
  ros::NodeHandle node;
  rl_agent::LaserScanMerger ls(node);

  while (ros::ok()) {
    ros::spin();
  }
  return 0;
};