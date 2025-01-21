#!/usr/bin/env python3
#
# MOBILE ROBOTS - FI-UNAM, 2025-1
# PATH FOLLOWING
#
# Instructions:
# Write the code necessary to move the robot along a given path.
# Consider a differential base. Max linear and angular speeds
# must be 0.8 and 1.0 respectively.
#

import rospy
import tf
import math
import numpy
from std_msgs.msg import Bool
from nav_msgs.msg import Path
from nav_msgs.srv import GetPlan, GetPlanRequest
from navig_msgs.srv import ProcessPath, ProcessPathRequest
from geometry_msgs.msg import Twist, PoseStamped, Pose, Point

NAME = "FULL NAME"

pub_goal_reached = None
pub_cmd_vel = None
loop        = None
listener    = None

def calculate_control(robot_x, robot_y, robot_a, goal_x, goal_y, alpha, beta, v_max, w_max):
    v,w = 0,0
    #
    # TODO:
    # Implement the control law given by:
    #
    # v = v_max*math.exp(-error_a*error_a/alpha)
    # w = w_max*(2/(1 + math.exp(-error_a/beta)) - 1)
    #
    # where error_a is the angle error
    # and v_max, w_max, alpha and beta, are tunning constants.
    # Remember to keep error angle in the interval (-pi,pi]
    # Return the tuple [v,w]
    #
        
    return [v,w]

def follow_path(path, alpha, beta, v_max, w_max):
    #
    # TODO:
    # Use the calculate_control function to move the robot along the path.
    # Path is given as a sequence of points [[x0,y0], [x1,y1], ..., [xn,yn]]
    # You can use the following steps to perform the path tracking:
    #
    # Set goal point as the first point of the path
    # Get robot position with Pr, robot_a = get_robot_pose()
    #
    # WHILE distance to last point > tol and not rospy.is_shutdown() 
    #     Calculate control signals v and w
    #     Publish the control signals with the function publish_twist()
    #     Get robot position
    #     If dist to goal point is less than 0.3 (you can change this constant)
    #         Change goal point to the next point in the path
    #
            
    return
        

def publish_twist(v,w):
    loop = rospy.Rate(20)
    msg = Twist()
    msg.linear.x = v
    msg.angular.z = w
    pub_cmd_vel.publish(msg)
    loop.sleep()
    
    
def callback_global_goal(msg):
    print("Calculating path from robot pose to " + str([msg.pose.position.x, msg.pose.position.y]))
    [robot_x, robot_y], robot_a = get_robot_pose()
    req = GetPlanRequest(goal=PoseStamped(pose=msg.pose))
    req.start.pose.position = Point(x=robot_x, y=robot_y)
    path = rospy.ServiceProxy('/path_planning/plan_path', GetPlan)(req).plan
    if len(path.poses) < 2:
        print("Cannot calculate path")
        return
    try:
        smooth_path = rospy.ServiceProxy('/path_planning/smooth_path',ProcessPath)(ProcessPathRequest(path=path)).processed_path
        path = smooth_path
    except:
        pass
    v_max = rospy.get_param("~v_max",0.8)
    w_max = rospy.get_param("~w_max",1.0)
    alpha = rospy.get_param("~alpha",1.0)
    beta  = rospy.get_param("~beta", 0.1)
    print("Following path with [v_max, w_max, alpha, beta]=" + str([v_max, w_max, alpha, beta]))
    follow_path([numpy.asarray([p.pose.position.x, p.pose.position.y]) for p in path.poses], alpha, beta, v_max, w_max)
    pub_cmd_vel.publish(Twist())
    pub_goal_reached.publish(True)
    print("Global goal point reached")

def get_robot_pose():
    try:
        ([x, y, z], [qx,qy,qz,qw]) = listener.lookupTransform('map', 'base_link', rospy.Time(0))
        return numpy.asarray([x, y]), 2*math.atan2(qz, qw)
    except:
        return numpy.asarray([0,0]),0

def main():
    global pub_cmd_vel, pub_goal_reached, loop, listener
    print("PATH FOLLOWING - " + NAME)
    rospy.init_node("path_follower")
    rospy.Subscriber('/move_base_simple/goal', PoseStamped, callback_global_goal)
    pub_cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    pub_goal_reached = rospy.Publisher('/navigation/goal_reached', Bool, queue_size=10)
    listener = tf.TransformListener()
    loop = rospy.Rate(10)
    print("Waiting for service for path planning...")
    rospy.wait_for_service('/path_planning/plan_path')
    print("Service for path planning is now available.")
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    
