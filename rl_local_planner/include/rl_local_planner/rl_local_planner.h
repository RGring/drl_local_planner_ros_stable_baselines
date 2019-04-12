/*
 * @name	 	  rl_local_planner.h
 * @brief	 	  Connector of move_base and rl_agent using RL library. Forwards action of rl_agent
 * 				    to move_base each time step.
 * @author  	Ronja Gueldenring
 * @date 	  	2019/04/05
 **/


#ifndef RL_LOCAL_PLANNER_H
#define RL_LOCAL_PLANNER_H
#include <ros/ros.h>
#include <math.h>
#include <nav_core/base_local_planner.h>
#include <std_msgs/Bool.h>
#include <geometry_msgs/Twist.h>
#include <rl_msgs/SetPath.h>


namespace rl_local_planner {

  /**
   * This class serves as connector between move\_base and the rl_agent, that uses 
   * a RL-library. The rl_agent provides an action each timestep that is forwarded
   * to move_base over this class.
   */
  class RLLocalPlanner : public nav_core::BaseLocalPlanner {
    public:
      /**
       * @brief  Constructor for RLLocalPlanner wrapper
       */
      RLLocalPlanner();

      
      virtual ~RLLocalPlanner() = default;


      /**
       * @brief  Constructs the ros wrapper
       * @param name The name to give this instance of the trajectory planner
       * @param tf A pointer to a transform listener
       * @param costmap The cost map to use for assigning costs to trajectories
       */
      void initialize(std::string name, tf::TransformListener* tf, costmap_2d::Costmap2DROS* costmap_ros);

      /**
       * @brief  Check if the goal pose has been achieved
       * @return True if achieved, false otherwise
       */
      bool isGoalReached();

      /**
       * @brief  Set the plan that the controller is following
       * @param orig_global_plan The plan to pass to the controller
       * @return True if the plan was updated successfully, false otherwise
       */
      bool setPlan(const std::vector<geometry_msgs::PoseStamped>& orig_plan);

      /**
       * @brief  Given the current position, orientation, and velocity of the robot, compute velocity commands to send to the base
       * @param cmd_vel Will be filled with the velocity command to be passed to the robot base
       * @return >0 if a valid velocity command was found, otherwise a result <0, contains the fail_code
       */
      bool computeVelocityCommands(geometry_msgs::Twist& cmd_vel);
    protected:
      ros::NodeHandle nh_;
    private:
      // Class variables
      tf::TransformListener* tf_;
      tf::Vector3 original_goal_;                             // goal of global plan
      geometry_msgs::Twist action_;                           // most recent action
      std::string path_frame_;                                // name of path frame           
      std::string robot_frame_;                               // name of robot frame
      double goal_threshold_;                                 // threshold, when goal is reached
      bool is_action_new_;                                    // True, if new action is available
      int rl_mode_;                                           // mode of rl_agent
      bool done_;                                             // True, if the rl_agent is done

      // Services
      ros::ServiceClient rl_agent_;
      ros::ServiceClient set_path_service_;
      
      // Publisher
      ros::Publisher trigger_agent_pub;

      // Subscriber
      ros::Subscriber agent_action_sub_;
      /**
       * @brief cmd_vel of rl-agent is published here.
       * @param cmd_vel velocity command
       */
      void agent_action_callback_(const geometry_msgs::Twist& cmd_vel);
      ros::Subscriber done_sub_;
      /**
       * @brief If episode is finished done=True is published here.
       * @param done True, if episode is done.
       */
      void done_callback_(const std_msgs::Bool& done);

      // functions
       /**
       * @brief Mean square distance.
       * @param x x-position
       * @param y x-position
       * @retur sqrt(x^2 + y^2)
       */
      double metric_dist(double x, double y);
  };
};
#endif /* RL_LOCAL_PLANNER_H */
