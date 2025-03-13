#!/usr/bin/env python3
#
# MOBILE ROBOTS - FI-UNAM, 2025-2
# PATH SMOOTHING BY GRADIENT DESCEND
#
# Instructions:
# Write the code necessary to smooth a path using the gradient descend algorithm.
# MODIFY ONLY THE SECTIONS MARKED WITH THE 'TODO' COMMENT
#

import numpy
import heapq
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import Pose, PoseStamped, Point
from navig_msgs.srv import ProcessPath
from navig_msgs.srv import ProcessPathResponse

NAME = "Velasco Vanegas Ricardo Alonso"

def smooth_path(Q, alpha, beta, max_steps):
    P = numpy.copy(Q)
    tol = 0.00001  # Tolerancia
    epsilon = 0.1  # Paso
    steps = 0 
    n = len(Q)  # puntos en el camino
    
    nabla = numpy.zeros(Q.shape) # inicializar en 0
    while steps < max_steps:
        prev_P = numpy.copy(P) 
        for i in range(1, n - 1):
            grad_smooth = 2 * P[i] - P[i - 1] - P[i + 1] #gradiente suavizado
            grad_attract = P[i] - Q[i]
            nabla[i] = alpha * grad_smooth + beta * grad_attract
        P -= epsilon * nabla
        if numpy.linalg.norm(P - prev_P) < tol: # revisar tolerancia si es menor
            break
       
        steps += 1  
    return P

def callback_smooth_path(req):
    global msg_smooth_path
    alpha = rospy.get_param('~alpha', 0.9)
    beta  = rospy.get_param('~beta', 0.1 )
    steps = rospy.get_param('~steps', 10000)
    print("Smoothing path with params: " + str([alpha,beta,steps]))
    start_time = rospy.Time.now()
    P = smooth_path(numpy.asarray([[p.pose.position.x, p.pose.position.y] for p in req.path.poses]), alpha, beta, steps)
    end_time = rospy.Time.now()
    print("Path smoothed after " + str(1000*(end_time - start_time).to_sec()) + " ms")
    msg_smooth_path.poses = []
    for i in range(len(req.path.poses)):
        msg_smooth_path.poses.append(PoseStamped(pose=Pose(position=Point(x=P[i,0],y=P[i,1]))))
    return ProcessPathResponse(processed_path=msg_smooth_path)

def main():
    global msg_smooth_path
    print("PATH SMOOTHING - " + NAME)
    rospy.init_node("path_smoothing", anonymous=True)
    rospy.Service('/path_planning/smooth_path', ProcessPath, callback_smooth_path)
    pub_path = rospy.Publisher('/path_planning/smooth_path', Path, queue_size=10)
    loop = rospy.Rate(1)
    msg_smooth_path = Path()
    msg_smooth_path.header.frame_id = "map"
    while not rospy.is_shutdown():
        pub_path.publish(msg_smooth_path)
        loop.sleep()

if __name__ == '__main__':
    main()
    
