/*
 * @name	 	image_generator.cpp
 * @brief	 	An image is generated from laserscan data and waypoints on the path
 * @author  	Ronja Gueldenring
 * @date 		2019/04/05
 **/
#include <rl_local_planner/image_generator.h>


namespace rl_image_generator {

	ImageGenerator::ImageGenerator(const ros::NodeHandle& node_handle)
							: nh_{node_handle}{

        // getting params from param server
        nh_.getParam("rl_agent/img_width_pos", img_width_pos_);
        nh_.getParam("rl_agent/img_width_neg", img_width_neg_);
        nh_.getParam("rl_agent/img_height", img_height_);
        nh_.getParam("rl_agent/resolution", resolution_);
        nh_.getParam("rl_agent/robot_frame", robot_frame_);

        //Services
        std::string img_service_name_ = ros::this_node::getName() + "/get_image";
        get_image_service_ = nh_.advertiseService(img_service_name_, &ImageGenerator::get_img_callback_, this);
	}

    bool ImageGenerator::get_img_callback_(rl_msgs::StateImageGenerationSrv::Request& request, rl_msgs::StateImageGenerationSrv::Response& response){
        nav_msgs::OccupancyGrid img = generate_image(request.scan, request.wps);
        response.img = img;
        return true;
    }//get_img_callback_

    nav_msgs::OccupancyGrid ImageGenerator::generate_image(sensor_msgs::LaserScan& scan, rl_msgs::Waypoint& wp){
        //Initializing image
        nav_msgs::OccupancyGrid img;
        img.header.stamp = ros::Time::now();
        img.header.frame_id = robot_frame_;
        img.info.resolution = resolution_;
        img.info.width = (img_width_pos_ + img_width_neg_);
        img.info.height = img_height_;
        img.info.origin.position.x = -img_width_neg_*resolution_;
        img.info.origin.position.y = -img_height_*resolution_/2.0;
        std::vector<int8_t> image;
        image.resize((img_height_*(img_width_pos_ + img_width_neg_)), 50);
        img.header.stamp = scan.header.stamp;

        //Adding laserscan infromation to the image
        add_scan_to_img_(image, scan);

        //Adding path infromation to the image
        if (wp.points.size() > 0){
            if(img.header.stamp > wp.header.stamp){
                img.header.stamp = wp.header.stamp;
            }
            geometry_msgs::Point zero_point;
            zero_point.x = 0.0;
            zero_point.y = 0.0;
            add_line_to_img_(image, zero_point.x, zero_point.y, wp.points[0].x, wp.points[0].y, 100);
            int num_wps = wp.points.size();
            for(int i = 1; i < num_wps; i++){
                add_line_to_img_(image, wp.points[i-1].x, wp.points[i-1].y, wp.points[i].x, wp.points[i].y, 100);
            }
        }
        img.data = image;
        return img;    
    }//generate_image

    void ImageGenerator::add_goal_point(std::vector<int8_t>& image, float x_goal, float y_goal){
        float size = 0.5;
        for (int iy = 0; iy < (int)(size/resolution_); iy++){
            for (int ix = 0; ix < (int)(size/resolution_); ix++){
                float y = y_goal - size/2 + iy*resolution_;
                float x = x_goal - size/2 + ix*resolution_;
                int index = point_to_index_(x,y);
                if(index >= 0 && index < (img_height_*(img_width_pos_ + img_width_neg_))){
                    image[index] = 100;
                }
            }
        }

    }//add_goal_point

    void ImageGenerator::add_scan_to_img_(std::vector<int8_t>& image, sensor_msgs::LaserScan& scan){
        if(scan.ranges.size() == 0){
            return;
        }
        tf::StampedTransform scan_transform_;
        try{
            tf_.lookupTransform(robot_frame_, scan.header.frame_id, 
                                ros::Time(0), scan_transform_);

        }
        catch (tf::TransformException ex){
            ROS_ERROR("%s",ex.what());
            ros::WallDuration(1.0).sleep();
        }

        double max_dist = sqrt(pow(((img_width_pos_ + img_width_neg_)*resolution_),2) + pow((img_height_*resolution_/2.0),2));
        for (int i=0; i < scan.ranges.size(); i+=1){
            // std::vector<double> angles = {scan.angle_min + i * scan.angle_increment - 0.333*scan.angle_increment,
            //                             scan.angle_min + i * scan.angle_increment - 0.0*scan.angle_increment,
            //                             scan.angle_min + i * scan.angle_increment - 0.333*scan.angle_increment
            // };
            std::vector<double> angles;
            angles.push_back(scan.angle_min + i * scan.angle_increment - 0.4444*scan.angle_increment);
            angles.push_back(scan.angle_min + i * scan.angle_increment - 0.3333*scan.angle_increment);
            angles.push_back(scan.angle_min + i * scan.angle_increment - 0.2222*scan.angle_increment);
            angles.push_back(scan.angle_min + i * scan.angle_increment - 0.1111*scan.angle_increment);
            angles.push_back(scan.angle_min + i * scan.angle_increment - 0.0*scan.angle_increment);
            angles.push_back(scan.angle_min + i * scan.angle_increment + 0.1111*scan.angle_increment);
            angles.push_back(scan.angle_min + i * scan.angle_increment + 0.2222*scan.angle_increment);
            angles.push_back(scan.angle_min + i * scan.angle_increment + 0.3333*scan.angle_increment);
            angles.push_back(scan.angle_min + i * scan.angle_increment + 0.4444*scan.angle_increment);
            for (int k=0; k < angles.size(); k++){
                double angle = angles[k]; //scan.angle_min + i * scan.angle_increment;
                double length;
                double length2;
                if (isnan(scan.ranges[i]) || scan.ranges[i] == 0.0){
                    continue;
                }else{
                    length = scan.ranges[i];
                    length2 = max_dist;
                }

                // Transform laserpoint to robot frame
                tf::Vector3 laser_point(cos(angle)*length, sin(angle)*length, 0.);
                tf::Vector3 laser_point_transformed = scan_transform_* laser_point;
                float x1 = laser_point_transformed.getX();
                float y1 = laser_point_transformed.getY();

                tf::Vector3 laser_point2(cos(angle)*length2, sin(angle)*length2, 0.);
                tf::Vector3 laser_point2_transformed = scan_transform_* laser_point2;
                float x2 = laser_point2_transformed.getX();
                float y2 = laser_point2_transformed.getY();
                add_line_to_img_(image, x1, y1, x2, y2, 0);
            }
        }
        return;
    }//add_scan_to_img

    void ImageGenerator::add_line_to_img_(std::vector<int8_t>& image, float x1, float y1, float x2, float y2, int value){
        const bool steep = (fabs(y2 - y1) > fabs(x2 - x1));
        if(steep)
        {
            std::swap(x1, y1);
            std::swap(x2, y2);
        }
        
        if(x1 > x2)
        {
            std::swap(x1, x2);
            std::swap(y1, y2);
        }
        
        const float dx = x2 - x1;
        const float dy = fabs(y2 - y1);
        
        float error = dx / 2.0f;
        const float ystep = (y1 < y2) ? resolution_*0.7 : -resolution_*0.7;
        float y = y1;
                    
        for(float x=x1; x<x2; x+=resolution_*0.7)
        {
            int index;
            if(steep)
            {
                index = point_to_index_(y ,x);
            }
            else
            {
                index = point_to_index_(x ,y);
            }

            if(index >= 0 && index < (img_height_*(img_width_pos_ + img_width_neg_))){
                image[index] = value;
            }
        
            error -= dy;
            if(error < 0)
            {
                y += ystep;
                error += dx;
            }
        }
        return;
    }//add_line_to_img_

    int ImageGenerator::point_to_index_(double x, double y){
            int y_index = (int) ceil((y + (((float)img_height_* resolution_))/2.0)/resolution_);
            int x_index = (int) ceil((x + (((float)img_width_neg_* resolution_)))/resolution_);

            int index = (y_index) * (img_width_pos_ + img_width_neg_) + (x_index);
            if (x_index < 0 || x_index >= (img_width_pos_ + img_width_neg_) || y_index < 0 || y_index > img_height_){
                index = (img_width_pos_ + img_width_neg_) * img_height_ + 1;
            }
            return index;
    }//point_to_index_

	double ImageGenerator::metric_dist(double x, double y){
		double dist = sqrt(pow(x , 2) + pow(y , 2));
		return dist; 
	} //metric_dist

    int ImageGenerator::get_img_neg_width(){
        return img_width_neg_;
    }

    int ImageGenerator::get_img_pos_width(){
        return img_width_pos_;
    }

    int ImageGenerator::get_img_height(){
        return img_height_;
    }

    float ImageGenerator::get_res(){
        return resolution_;
    }

}; // namespace rl_image_generator


int main(int argc, char** argv){
    ros::init(argc, argv, "image_generator");
    ros::NodeHandle node;
    rl_image_generator::ImageGenerator ig(node);
    ros::WallRate r(100);
    while (ros::ok()) {
        ros::spinOnce();
        r.sleep();
    }
    return 0;
};