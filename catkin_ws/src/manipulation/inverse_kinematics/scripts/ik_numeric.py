#!/usr/bin/env python3
#
# MOBILE ROBOTS - FI-UNAM, 2025-2
# INVERSE KINEMATICS USING NEWTON-RAPHSON
#
# Instructions:
# Calculate the inverse kinematics using
# the Newton-Raphson method for root finding.
# Modify only sections marked with the 'TODO' comment
#
import math
import sys
import rospy
import numpy
import tf
import tf.transformations as tft
import urdf_parser_py.urdf
from std_msgs.msg import Float64MultiArray
from manip_msgs.srv import *
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

prompt = ""
NAME = "MURILLO SANTOS JAVIER EDUARDO"
   
def forward_kinematics(q, T, W):
    H = tft.identity_matrix()
    
    for i in range(7):
        R = tft.rotation_matrix(q[i], W[i])
        H = tft.concatenate_matrices(H, T[i], R)

    H = tft.concatenate_matrices(H, T[7])
    
    x, y, z = H[0, 3], H[1, 3], H[2, 3]
    R, P, Y = tft.euler_from_matrix(H)
    
    return numpy.asarray([x, y, z, R, P, Y])


def jacobian(q, T, W):
    delta_q = 1e-6  # Pequeña perturbación
    J = numpy.zeros((6, 7))  # Jacobiano de 6 filas (x,y,z,R,P,Y) por 7 articulaciones

    for i in range(7):
        q_forward = numpy.copy(q)
        q_backward = numpy.copy(q)

        q_forward[i] += delta_q
        q_backward[i] -= delta_q

        fk_plus = forward_kinematics(q_forward, T, W)
        fk_minus = forward_kinematics(q_backward, T, W)
        diff = (fk_plus - fk_minus) / (2 * delta_q)
        for j in range(3, 6):
            diff[j] = (diff[j] + numpy.pi) % (2 * numpy.pi) - numpy.pi

        J[:, i] = diff

    return J


def inverse_kinematics(x, y, z, roll, pitch, yaw, T, W, init_guess=numpy.zeros(7), max_iter=20):
    pd = numpy.asarray([x, y, z, roll, pitch, yaw])
    q = numpy.copy(init_guess)
    iterations = 0
    TOL = 1e-4

    while iterations < max_iter:
        p = forward_kinematics(q, T, W)
        error = p - pd

        for j in range(3, 6):
            error[j] = (error[j] + numpy.pi) % (2 * numpy.pi) - numpy.pi

        if numpy.linalg.norm(error) < TOL:
            break

        J = jacobian(q, T, W)
        dq = numpy.linalg.pinv(J).dot(error)
        q = q - dq
        q = (q + numpy.pi) % (2 * numpy.pi) - numpy.pi

        iterations += 1

    success = iterations < max_iter and angles_in_joint_limits(q)
    print(prompt + f"IK iterations: {iterations}")
    return success, q, iterations

   
def get_polynomial_trajectory_multi_dof(Q_start, Q_end, duration=1.0, time_step=0.05):
    clt = rospy.ServiceProxy("/manipulation/polynomial_trajectory", GetPolynomialTrajectory)
    req = GetPolynomialTrajectoryRequest()
    req.p1 = Q_start
    req.p2 = Q_end
    req.duration = duration
    req.time_step = 0.05
    resp = clt(req)
    Q = []
    T = []
    for p in resp.trajectory.points:
        Q.append(p.positions)
        T.append(p.time_from_start.to_sec())
    return numpy.asarray(Q), numpy.asarray(T)

def get_model_info(joint_names):
    robot_model = urdf_parser_py.urdf.URDF.from_parameter_server()
    joints = []
    transforms = []
    for name in joint_names:
        for joint in robot_model.joints:
            if joint.name == name:
                joints.append(joint)
    for joint in joints:
        T = tft.translation_matrix(joint.origin.xyz)
        R = tft.euler_matrix(joint.origin.rpy[0], joint.origin.rpy[1], joint.origin.rpy[2])
        transforms.append(tft.concatenate_matrices(T,R))
    return joints, transforms

def angles_in_joint_limits(q):
    for i in range(len(q)):
        if q[i] < joints[i].limit.lower or q[i] > joints[i].limit.upper:
            print(prompt+"Articular position out of joint bounds")
            return False
    return True

def callback_forward_kinematics(req):
    if len(req.q) != 7:
        print(prompt+"By the moment, only 7-DOF arm is supported")
        return False
    resp = ForwardKinematicsResponse()
    W = [joints[i].axis for i in range(len(joints))]  
    resp.x,resp.y,resp.z,resp.roll,resp.pitch,resp.yaw = forward_kinematics(req.q, transforms, W)
    return resp

def get_trajectory_time(p1, p2, speed_factor):
    p1 = numpy.asarray(p1)
    p2 = numpy.asarray(p2)
    m = max(numpy.absolute(p1 - p2))
    return m/speed_factor + 0.5

def callback_ik_for_trajectory(req):
    global max_iterations
    Pd = [req.x, req.y, req.z, req.roll, req.pitch, req.yaw]
    print(prompt+"Calculating IK and trajectory for " + str(Pd))
    if len(req.initial_guess) <= 0 or req.initial_guess == None:
        initial_guess = rospy.wait_for_message("/hardware/arm/current_pose", Float64MultiArray)
        initial_guess = initial_guess.data
    else:
        initial_guess = req.initial_guess
    W = [joints[i].axis for i in range(len(joints))]  
    p1 = forward_kinematics(initial_guess, transforms, W)
    p2 = [req.x, req.y, req.z, req.roll, req.pitch, req.yaw]
    t  = req.duration if req.duration > 0 else get_trajectory_time(p1, p2, 0.25)
    dt = req.time_step if req.time_step > 0 else 0.05
    X,T = get_polynomial_trajectory_multi_dof(p1, p2, duration=t, time_step=dt)
    print(prompt + f"Trajectory: {len(T)} points over {t:.2f} seconds")
    trj = JointTrajectory()
    trj.header.stamp = rospy.Time.now()
    q = initial_guess
    for i in range(len(X)):
        x, y, z, roll, pitch, yaw = X[i]
        success, q, iterations = inverse_kinematics(x, y, z, roll, pitch, yaw, transforms, W, q, max_iterations)
        print(prompt + "Estoy dentro de la función callback_ik_for_trajectory")
        print(prompt + f"[STEP {i+1}] Iterations: {iterations}")

        if not success:
            return False
        p = JointTrajectoryPoint()
        p.positions = q
        p.time_from_start = rospy.Duration.from_sec(T[i])
        trj.points.append(p)
    resp = InverseKinematicsPose2TrajResponse()
    resp.articular_trajectory = trj
    return resp
    
        
def callback_ik_for_pose(req):
    global max_iterations
    [x,y,z,R,P,Y] = [req.x,req.y,req.z,req.roll,req.pitch,req.yaw]
    print(prompt+"Calculating inverse kinematics for pose: " + str([x,y,z,R,P,Y]))
    if len(req.initial_guess) <= 0 or req.initial_guess == None:
        init_guess = rospy.wait_for_message("/hardware/arm/current_pose", Float64MultiArray, 5.0)
        init_guess = initial_guess.data
    else:
        init_guess = req.initial_guess
    resp = InverseKinematicsPose2PoseResponse()
    success, q, _ = inverse_kinematics(x, y, z, R, P, Y, transforms, W, init_guess, max_iterations)
    if not success:
        return False
    resp.q = q
    return resp        
    

def main():
    global joint_names, max_iterations, joints, transforms, prompt
    print("INITIALIZING INVERSE KINEMATIC NODE - " + NAME)
    rospy.init_node("ik_geometric")
    prompt = rospy.get_name().upper() + ".->"
    joint_names    = rospy.get_param("~joint_names", [])
    max_iterations = rospy.get_param("~max_iterations", 20)
    print(prompt+"Joint names: " + str(joint_names))
    print(prompt+"max_iterations: " + str(max_iterations))

    joints, transforms = get_model_info(joint_names)
    if not (len(joints) > 6 and len(transforms) > 6):
        print("Inverse kinematics.->Cannot get model info from parameter server")
        sys.exit(-1)

    rospy.Service("/manipulation/forward_kinematics"   , ForwardKinematics, callback_forward_kinematics)    
    rospy.Service("/manipulation/ik_trajectory"        , InverseKinematicsPose2Traj, callback_ik_for_trajectory)
    rospy.Service("/manipulation/ik_pose"              , InverseKinematicsPose2Pose, callback_ik_for_pose)
    #loop = rospy.Rate(10)
    loop = rospy.Rate(40)
    while not rospy.is_shutdown():
        loop.sleep()

if __name__ == '__main__':
    main()


