#!/usr/bin/env python3
#
# MOBILE ROBOTS - FI-UNAM, 2025-2
# SPLIT AND MERGE ALGORITHM
#
# Instructions:
# Test different parameteres and compare the results.
# Parameters:
# 
#

import rospy
import numpy
import math
import tf
from sensor_msgs.msg   import LaserScan
from geometry_msgs.msg import Twist, Point
from visualization_msgs.msg import Marker

NAME = "WRITE HERE YOUR FULL NAME"

def adjust_line(points):
    [xm,ym] = numpy.mean(points, 0)
    n,d = 0,0
    for x,y in points:
        n += (xm - x)*(ym - y)
        d += (ym - y)**2 - (xm - x)**2
    theta = 0.5*math.atan2(-2*n , d)
    rho   = xm*math.cos(theta) + ym*math.sin(theta)
    length= numpy.linalg.norm(points[0] - points[-1])
    return rho, theta, xm, ym, length

def find_farthest_point(points, rho, theta):
    distances = [abs(points[i][0]*math.cos(theta) + points[i][1]*math.sin(theta) - rho) for i in range(len(points))]
    idx = numpy.argmax(distances)
    return idx, distances[idx]
        
def split(points, threshold, min_points):
    lines = []
    #
    # TODO:
    # Implement the 'split' part of the split and merge algorithm for finding lines.
    # Implement the recursive method of the algorithm. 
    #
    
    return lines


def merge(lines, rho_tol, theta_tol):
    new_lines = []
    #
    # TODO:
    # Implement the 'merge' part of the split and merge algorithm.
    # Two segments are merged into one if rho and theta differences
    # are both smaller than a tolerance.
    #
    
    return new_lines
            
def split_and_merge(points, threshold, min_points, rho_tol, theta_tol):
    lines = split(points, threshold, min_points)
    lines = merge(lines, rho_tol, theta_tol)
    return lines

def get_line_markers(lines):
    marker = Marker()
    marker.header.frame_id = "map";
    marker.header.stamp = rospy.Time.now();
    marker.ns = "segmented_lines";
    marker.id = 7;
    marker.type = Marker.LINE_LIST;
    marker.action = Marker.ADD;
    marker.scale.x = 0.1;
    marker.pose.orientation.w = 1.0
    marker.color.a = 1.0;
    marker.color.r = 0.0;
    marker.color.g = 0.0;
    marker.color.b = 1.0;
    for [rho, theta, xm, ym, length] in lines:
        a  = math.cos(theta)
        b  = math.sin(theta)
        marker.points.append(Point(xm + length/2*(-b), ym + length/2*(a), 0.5))
        marker.points.append(Point(xm - length/2*(-b), ym - length/2*(a), 0.5))
    return marker

def get_laser_pose():
    global listener
    try:
        ([x, y, z], rot) = listener.lookupTransform('map', 'laser_link', rospy.Time(0))
        a = 2*math.atan2(rot[2], rot[3])
        a = a - 2*math.pi if a > math.pi else a
        return [x, y, z, a]
    except:
        pass
    return [0,0,0,0]

def callback_scan(msg):
    global pub_line_markers, distance_threshold, min_points_counting, rho_tolerance, theta_tolerance
    x,y,z,a = get_laser_pose()
    points = []
    for i in range(len(msg.ranges)):
        if not (math.isnan(msg.ranges[i]) or msg.ranges[i] >= msg.range_max):
            r,theta = msg.ranges[i], i*msg.angle_increment + msg.angle_min + a
            points.append([r*math.cos(theta) + x, r*math.sin(theta) + y])
    points = numpy.asarray(points)
    lines = split_and_merge(points, distance_threshold, min_points_counting, rho_tolerance, theta_tolerance)
    pub_line_markers.publish(get_line_markers(lines))
    return

def main():
    global pub_line_markers, listener, distance_threshold, min_points_counting, rho_tolerance, theta_tolerance
    print("SPLIT AND MERGE ALGORITHM - " + NAME)
    rospy.init_node("split_and_merge")
    rospy.Subscriber("/hardware/scan", LaserScan, callback_scan)
    pub_line_markers = rospy.Publisher("/navigation/segmented_lines_marker", Marker, queue_size=1)
    listener = tf.TransformListener()
    loop = rospy.Rate(10)

    #
    # TODO:
    # Modify the following parameters and compare the results:
    #
    distance_threshold  = rospy.get_param("~dist", 0.1)     #Distance threshold to consider a point as part of a candidate line. 
    min_points_counting = rospy.get_param("~points", 1)     #Minimum number of points a line should contain.
    rho_tolerance       = rospy.get_param("~rho", 0.05)     #RHO and THETA error tolerance to consider two lines as one.
    theta_tolerance     = rospy.get_param("~theta", 0.05)
    print("Trying to find lines with parameters:")
    print("Distance threshold: " + str(distance_threshold))
    print("Min points per line: " + str(min_points_counting))
    print("Rho tolerance: " + str(rho_tolerance))
    print("Theta tolerance: " + str(theta_tolerance))

    while not rospy.is_shutdown():
        loop.sleep()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    
