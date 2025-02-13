#!/usr/bin/env python3
#
# MOBILE ROBOTS - FI-UNAM, 2025-2
# THE PLATFORM ROS 
#
# Instructions:
# Write a program to move the robot forwards until the laser
# detects an obstacle in front of it.
# Required publishers and subscribers are already declared and initialized.
#

import rospy
from sensor_msgs.msg   import LaserScan
from geometry_msgs.msg import Twist

NAME = "Jessica Stephanie Garcia Monjaraz"

def callback_scan(msg):
    global obstacle_detected
    #
    # TODO:
    # Se calcula el índice central del arreglo de las mediciones de angulos que toma el LiDAR, la mitad sería el frente del robot. 
    n = int((msg.angle_max - msg.angle_min) / msg.angle_increment / 2)
    # Se detecta un obstáculo basado en la distancia umbral (1m), dará un booleano, es decir, si hay un objeto o no lo hay. 
    obstacle_detected = msg.ranges[n] < 1.0
    
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
        # Se crea el mensaje Twist (msg)
        msg_cmd_vel = Twist()
        # Define la velocidad, si hay un obstaculo, no se mueve (0), si no hay obstáculo, avanza (0.3)
        msg_cmd_vel.linear.x = 0 if obstacle_detected else 0.3
        # Use the 'obstacle_detected' variable to check if there is an obstacle, (lo que se hizo en la linea 43). 
        # Publica el mensaje de velocidad. (msg_cmd_vel)
        pub_cmd_vel.publish(msg_cmd_vel)
        
        loop.sleep()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    
