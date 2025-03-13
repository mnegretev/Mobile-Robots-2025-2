#!/usr/bin/env python3
#
# MOBILE ROBOTS - FI-UNAM, 2025-2
# OBSTACLE AVOIDANCE BY POTENTIAL FIELDS
#
# Instructions:
# Complete the code to implement obstacle avoidance by potential fields
# using the attractive and repulsive fields technique.
# Tune the constants alpha and beta to get a smooth movement. 
#

import rospy
import tf
import math
import numpy
from geometry_msgs.msg import Twist, PoseStamped, Point, Vector3
from visualization_msgs.msg import Marker
from sensor_msgs.msg import LaserScan

listener    = None
pub_cmd_vel = None
pub_markers = None
laser_readings = None
v_max = 0.6
w_max = 1.0

NAME = "JUAN SALVADOR PACHECO JARILLO"

def calculate_control(goal_x, goal_y, alpha, beta):
    v,w = 0,0
    #
    # TODO:
    # Implement the control law described in the slides.
    # Consider that goal point is given w.r.t. robot, i.e., robot is always at zero.
    # Return v and w as a tuble [v,w]
    # 
    error_a = math.atan2(goal_y, goal_x)
    # Error de àngulo acotado al intervalo (-pi , pi]
    error_a = (error_a+math.pi)%(2*math.pi) - math.pi
    
    v = v_max*math.exp(-error_a*error_a/alpha)
    w = w_max*(2/(1 + math.exp(-error_a/beta)) - 1)
   
    # where error_a is the angle error
    # and v_max, w_max, alpha and beta, are tunning constants.
    # Remember to keep error angle in the interval (-pi,pi]
    # Return the tuple [v,w]
    #  
    
    return [v,w]

def attraction_force(goal_x, goal_y, eta):
    force_x, force_y = 0,0 #PREGUNTAR AL PROFE SOBRE ESTO
    #
    # TODO:
    # Calculate the attraction force, given the goal positions
    # w.r.t. robot's frame, i.e., robot is always at zero.
    # Return a tuple of the form [force_x, force_y]
    # where force_x and force_y are the X and Y components
    # of the resulting attraction force w.r.t. robot
    #ALGORITMO
    #norm_qg = numpy.linalg.norm(numpy.array([goal_x, goal_y]))
    #if norm_qg == 0:  #Checa si el resultado es cero.
    #    return numpy.array([0, 0])
        
    #force_att = -eta*(numpy.array([goal_x, goal_y])/norm_qg)
    
    force_x = -eta*(goal_x) / math.sqrt(goal_x**2 + goal_y**2)
    force_y = -eta*(goal_y) / math.sqrt(goal_x**2 + goal_y**2)
    
    return numpy.asarray([force_x, force_y])



def rejection_force(laser_readings, zeta, d0):
    N = len(laser_readings)
    if N == 0:
        return [0, 0]
    force_x, force_y = 0, 0
    #
    # TODO:
    # Calculate the total rejection force given by the average
    # of the rejection forces caused by each laser reading.
    # laser_readings is an array where each element is a tuple [distance, angle]
    # both measured w.r.t. robot's frame.
    # See lecture notes for equations to calculate rejection forces.
    # Return a tuple of the form [force_x, force_y]
    # where force_x and force_y are the X and Y components
    # of the resulting rejection force
    #
    for [d, theta] in laser_readings:
        if d < d0:
            rho = zeta*math.sqrt((1/d)-(1/d0))
        else:
            rho = 0
        force_x = force_x + rho*math.cos(theta)
        force_y = force_y + rho*math.sin(theta)
    
    force_x = force_x/N
    force_y = force_y/N
    
    
    return numpy.asarray([force_x, force_y])

def move_by_pot_fields(global_goal_x, global_goal_y, epsilon, tol, eta, zeta, d0, alpha, beta):
    #
    # TODO
    # Implement potential fields given a goal point and tunning constants.
    # You can use the following steps:
    #
    # Get the goal point w.r.t. robot by calling the get_goal_point_wrt_robot function.
    # WHILE distance to goal is greater than a tolerance AND not rospy.is_shutdown():
    #    Calculate the attraction force Fa (call the corresponding function)
    #    Calculate the rejection force Fr (call the corresponding function)
    #    Calculate the resulting force F = Fa + Fr
    #    Calculate the next position P the robot should move to, using gradient descend: P = -epsilon*F
    #    Calculate the control laws v,w to move the robots towards P
    #    Call the function publish_speed_and_forces(v, w, Fa, Fr, F) (moves the robot and displays forces)
    #    Get the goal point w.r.t. robot by calling the get_goal_point_wrt_robot function.
    
    Pg = get_goal_point_wrt_robot(global_goal_x, global_goal_y)
    
    while numpy.linalg.norm(Pg) > tol and not rospy.is_shutdown():
        Fa = attraction_force(Pg[0], Pg[1], eta)
        Fr = rejection_force(laser_readings, zeta, d0)
        F = Fa + Fr
        P = -epsilon*F
        v, w = calculate_control(Pg[0], Pg[1], alpha, beta)
        
        publish_speed_and_forces(v, w, Fa, Fr, F)
        Pg = get_goal_point_wrt_robot(global_goal_x, global_goal_y)
   
         
        

def get_goal_point_wrt_robot(goal_x, goal_y):
    robot_x, robot_y, robot_a = get_robot_pose(listener)
    delta_x = goal_x - robot_x
    delta_y = goal_y - robot_y
    goal_x =  delta_x*math.cos(robot_a) + delta_y*math.sin(robot_a)
    goal_y = -delta_x*math.sin(robot_a) + delta_y*math.cos(robot_a)
    return [goal_x, goal_y]

def get_robot_pose(listener):
    try:
        ([x, y, z], [qx,qy,qz,qw]) = listener.lookupTransform('map', 'base_link', rospy.Time(0))
        return [x, y, 2*math.atan2(qz, qw)]
    except:
        return [0,0,0]

def publish_speed_and_forces(v, w, Fa, Fr, F):
    loop = rospy.Rate(20)
    pub_cmd_vel.publish(Twist(linear=Vector3(x=v), angular=Vector3(z=w)))
    pub_markers.publish(get_force_marker(Fa[0], Fa[1], [0.0, 0.0, 1.0, 1.0], 0))
    pub_markers.publish(get_force_marker(Fr[0], Fr[1], [1.0, 0.0, 0.0, 1.0], 1))
    pub_markers.publish(get_force_marker(F [0], F [1], [0.0, 0.6, 0.0, 1.0], 2))
    loop.sleep()

def get_force_marker(force_x, force_y, color, id):
    mrk = Marker()
    mrk.header.frame_id = "base_link"
    mrk.header.stamp = rospy.Time.now()
    mrk.ns = "pot_fields"
    mrk.id = id
    mrk.type = Marker.ARROW
    mrk.action = Marker.ADD
    mrk.pose.orientation.w = 1
    mrk.color.r, mrk.color.g, mrk.color.b, mrk.color.a = color
    mrk.scale.x, mrk.scale.y, mrk.scale.z = [0.07, 0.1, 0.15]
    mrk.points.append(Point(x=0, y=0))
    mrk.points.append(Point(x=-force_x, y=-force_y))
    return mrk

def callback_scan(msg):
    global laser_readings
    laser_readings = [[msg.ranges[i], msg.angle_min+i*msg.angle_increment] for i in range(len(msg.ranges))]

def callback_pot_fields_goal(msg):
    [goal_x, goal_y] = [msg.pose.position.x, msg.pose.position.y]
    print("Moving to goal point " + str([goal_x, goal_y]) + " by potential fields"    )
    epsilon = rospy.get_param('~epsilon', 0.5)
    tol     = rospy.get_param('~tol', 0.5)
    eta     = rospy.get_param('~eta', 2.0)
    zeta    = rospy.get_param('~zeta', 6.0)
    d0      = rospy.get_param('~d0', 1.0)
    alpha   = rospy.get_param('~alpha', 0.5)
    beta    = rospy.get_param('~beta', 0.5)
    move_by_pot_fields(goal_x, goal_y, epsilon, tol, eta, zeta, d0, alpha, beta)
    pub_cmd_vel.publish(Twist())
    print("Global goal point reached")

def main():
    global listener, pub_cmd_vel, pub_markers
    print("POTENTIAL FIELDS - " + NAME)
    rospy.init_node("pot_fields")
    rospy.Subscriber("/hardware/scan", LaserScan, callback_scan)
    rospy.Subscriber('/move_base_simple/goal', PoseStamped, callback_pot_fields_goal)
    pub_cmd_vel = rospy.Publisher('/cmd_vel', Twist,  queue_size=10)
    pub_markers = rospy.Publisher('/navigation/pot_field_markers', Marker, queue_size=10)
    listener = tf.TransformListener()
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    
