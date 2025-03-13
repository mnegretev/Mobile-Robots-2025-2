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

NAME = "Miguel Angel Ruiz Sànchez"

def calculate_control(goal_x, goal_y, alpha, beta):
    v,w = 0,0
    #
    # TODO:
    error_a = math.atan2(goal_y, goal_x)
    error_a = (error_a + math.pi) % (2 * math.pi) - math.pi

    v = v_max*math.exp(-error_a*error_a/alpha)
    w = w_max*(2/(1 + math.exp(-error_a/beta)) - 1)
    
    return [v,w]

def attraction_force(goal_x, goal_y, eta):
    force_x, force_y = 0,0 #PREGUNTAR AL PROFE SOBRE ESTO
    #
    # TODO:
    q_g = numpy.array([goal_x,goal_y])
    mod_q_g = numpy.linalg.norm(q_g)

    if mod_q_g != 0:
        force_x = -eta * (q_g[0] / mod_q_g)
        force_y = -eta * (q_g[1] / mod_q_g)

    return numpy.asarray([force_x, force_y])

def rejection_force(laser_readings, zeta, d0):
    N = len(laser_readings)
    if N == 0:
        return [0, 0]
    force_x, force_y = 0, 0
    #
    # TODO:
    for d,an in laser_readings:
        if d < d0:
            rho = zeta * numpy.sqrt((1/d) - (1/d0))
        else: 
            rho = 0
        force_x += rho * numpy.cos(an)
        force_y += rho * numpy.sin(an)

    force_x = (force_x/N)
    force_y = (force_y/N)

    return numpy.asarray([force_x, force_y])

def move_by_pot_fields(global_goal_x, global_goal_y, epsilon, tol, eta, zeta, d0, alpha, beta):
    #
    # TODO
    p_g = get_goal_point_wrt_robot(global_goal_x,global_goal_y)
    while numpy.linalg.norm(p_g) > tol and not rospy.is_shutdown():
        force_a = attraction_force(p_g[0],p_g[1],eta)
        force_r = rejection_force(laser_readings, zeta, d0)
        F = force_a + force_r
        P = -epsilon*F
        v,w = calculate_control(P[0],P[1], alpha, beta)
        publish_speed_and_forces(v, w, force_a, force_r, F)
        p_g = get_goal_point_wrt_robot(global_goal_x,global_goal_y)
    return
        

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
    
