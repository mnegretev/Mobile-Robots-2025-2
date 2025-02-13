#!/usr/bin/env python3
#
# MOBILE ROBOTS - FI-UNAM, 2025-2
# THE PLATFORM ROS 
#
# Instructions:
# Write a program to move the robot forwards until the laser
# detects an obstacle in front of it.
# Required publishers and subscribers are already declared and initialized.

import rospy
from sensor_msgs.msg   import LaserScan
from geometry_msgs.msg import Twist

NAME = "Lujan Perez Carlos Eduardo"

def callback_scan(msg):
    global obstacle_detected
    #
    # TODO:
    # Do something to detect if there is an obstacle in front of the robot.
    # Set the 'obstacle_detected' variable with True or False, accordingly.
    #
    n = int((msg.angle_max−msg.angle_min)/msg.angle_increment/2)
    obstacle_detected = msg.ranges[ n ] < 1.0
    return


def main():
    print("ROS BASICS - " + NAME)
    rospy.init_node("ros_basics")
    rospy.Subscriber("/hardware/scan", LaserScan, callback_scan)
    pub_cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    loop = rospy.Rate(10)
    
    global obstacle_detected
    obstacle_detected = False
    while not rospy.is_shutdown():
        #
        # TODO:
        # Declare a Twist message and assign the appropiate speeds:
        # Move forward if there is no obstacle in front of the robot, and stop otherwise.
        # Use the 'obstacle_detected' variable to check if there is an obstacle. 
        # Publish the Twist message using the already declared publisher 'pub_cmd_vel'.
        msg_cmd_vel = Twist()
        msg_cmd_vel.linear.x = 0 if obstacle_detected else 0.3
        pub_cmd_vel.publish(msg_cmd_vel) 
      
        loop.sleep()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    
