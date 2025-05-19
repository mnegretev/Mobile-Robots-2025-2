#!/usr/bin/env python3
#
# MOBILE ROBOTS - FI-UNAM, 2025-2
# TRAJECTORY PLANNING BY POLYNOMIALS
#
# Instructions:
# Complete the code to calculate a trajectory given an initial and final
# position, velocity and acceleration, using a fifth-degree polynomial.
# Modify only sections marked with the 'TODO' comment
#
import math
import sys
import rospy
import numpy
import tf
import tf.transformations as tft
from std_msgs.msg import Float64MultiArray
from manip_msgs.srv import *
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

prompt = ""
NAME = "JOSÉ AUGUSTO ARENAS HERNÁNDEZ"

def get_polynomial_trajectory(q0, q1, dq0=0, dq1=0, ddq0=0, ddq1=1, t=1.0, step=0.05):
    T = numpy.arange(0, t, step)
    Q = numpy.zeros(T.shape)
    #
    # TODO:
    # Calculate a trajectory Q as a set of N values q using a
    # fifth degree polynomial.
    # Initial and final positions, velocities and accelerations
    # are given by q, dq, and ddq.
    # Trajectory must have a duration 't' and a sampling time 'step'
    # Return both the time T and position Q vectors 
    #

    # Coeficientes del polinomio de 5° grado:
    # q(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
    M = numpy.array([
        [1, 0,    0,     0,      0,       0],
        [0, 1,    0,     0,      0,       0],
        [0, 0,    2,     0,      0,       0],
        [1, t,  t**2,  t**3,   t**4,    t**5],
        [0, 1,  2*t,  3*t**2, 4*t**3,  5*t**4],
        [0, 0,   2,   6*t,   12*t**2, 20*t**3]
    ])
    b = numpy.array([q0, dq0, ddq0, q1, dq1, ddq1])
    a = numpy.linalg.solve(M, b)  # Resuelve sistema lineal para coeficientes

    for i in range(len(T)):
        ti = T[i]
        Q[i] = a[0] + a[1]*ti + a[2]*ti**2 + a[3]*ti**3 + a[4]*ti**4 + a[5]*ti**5

    return T, Q

    
def get_polynomial_trajectory_multi_dof(Q_start, Q_end, Qp_start=[], Qp_end=[],
                                        Qpp_start=[], Qpp_end=[], duration=1.0, time_step=0.05):
    Q = []
    T = []
    if len(Qp_start) == 0:
        Qp_start = numpy.zeros(len(Q_start))
    if len(Qpp_start) == 0:
        Qpp_start = numpy.zeros(len(Q_start))
    if len(Qp_end) == 0:
        Qp_end = numpy.zeros(len(Q_end))
    if len(Qpp_end) == 0:
        Qpp_end = numpy.zeros(len(Q_end))
    for i in range(len(Q_start)):
        T, Qi = get_polynomial_trajectory(Q_start[i], Q_end[i], Qp_start[i], Qp_end[i],
                                          Qpp_start[i], Qpp_end[i], duration, time_step)
        Q.append(Qi)
    Q = numpy.asarray(Q)
    Q = Q.transpose()
    return Q,T


def get_trajectory_time(p1, p2, speed_factor):
    p1 = numpy.asarray(p1)
    p2 = numpy.asarray(p2)
    m = max(numpy.absolute(p1 - p2))
    return m/speed_factor + 0.5


def callback_polynomial_trajectory(req):
    print(prompt+"Calculating polynomial trajectory")
    t  = req.duration if req.duration > 0 else get_trajectory_time(req.p1, req.p2, 0.25)
    Q, T = get_polynomial_trajectory_multi_dof(req.p1, req.p2, req.v1, req.v2, req.a1, req.a2, t, req.time_step)
    trj = JointTrajectory()
    trj.header.stamp = rospy.Time.now()
    for i in range(len(Q)):
        p = JointTrajectoryPoint()
        p.positions = Q[i]
        p.time_from_start = rospy.Duration.from_sec(T[i])
        trj.points.append(p)
    resp = GetPolynomialTrajectoryResponse()
    resp.trajectory = trj
    return resp 

def main():
    global joint_names, max_iterations, joints, transforms, prompt
    print("INITIALIZING TRAJECTORY PLANNER NODE - " + NAME)
    rospy.init_node("trajectory_planner")
    prompt = rospy.get_name().upper() + ".->"
    rospy.Service("/manipulation/polynomial_trajectory", GetPolynomialTrajectory, callback_polynomial_trajectory)
    loop = rospy.Rate(10)
    while not rospy.is_shutdown():
        loop.sleep()

if __name__ == '__main__':
    main()


