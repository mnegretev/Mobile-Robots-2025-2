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
import os
import sys
import rospy
import numpy
import tf
import tf.transformations as tft
import urdf_parser_py.urdf
from std_msgs.msg import Float64MultiArray
from manip_msgs.srv import *
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import csv

prompt = ""
NAME = "Garcia Monjaraz Jessica Stephanie"


   
def forward_kinematics(q, T, W):
    x, y, z, R, P, Y = 0, 0, 0, 0, 0, 0
    #
    # Calculate the forward kinematics given the set of seven angles 'q'
    # You can use the following steps:
    #     H = I   # Assign to H a 4x4 identity matrix
    #     for all qi in q:
    #         H = H * Ti * Ri
    #     H = H * Ti[7]
    #     Get xyzRPY from the resulting Homogeneous Transformation 'H'
    # Where:
    #     Ti are the Homogeneous Transforms from frame i to frame i-1 when joint i is at zero position
    #     Ri are the Homogeneous Transforms with zero-translation and rotation qi around axis Wi.
    #     Ti[7] is the final Homogeneous Transformation from gripper center to joint 7.
    # Hints:
    #     Use the tft.identity_matrix() function to get the 4x4 I
    #     Use the tft.concatenate_matrices() function for multiplying Homogeneous Transformations
    #     Use the tft.rotation_matrix() matrices Ri.
    #     Use the tft.euler_from_matrix() function to get RPY from matrix H
    #     Check online documentation of these functions:
    #     http://docs.ros.org/en/jade/api/tf/html/python/transformations.html
    #
    
    # Initialize H as a 4x4 identity matrix
    H = tft.identity_matrix()
    
    # Iterate through each joint angle qi
    for i in range(len(q)):
        # Compute the rotation matrix Ri for qi around axis Wi
        Ri = tft.rotation_matrix(q[i], W[i])
        # Update H by multiplying it with Ti and Ri
        H = tft.concatenate_matrices(H, T[i], Ri)
    
    # Multiply H with the final transformation Ti[7]
    H = tft.concatenate_matrices(H, T[7])
    
    # Extract x, y, z and RPY from the resulting Homogeneous Transformation H
    x, y, z = H[0:3, 3]
    R, P, Y = tft.euler_from_matrix(H)
    
    return numpy.asarray([x, y, z, R, P, Y])

def jacobian(q, T, W):
    delta_q = 0.000001
    #
    # Calculate the Jacobian given a kinematic description Ti and Wi
    # where:
    # Ti are the Homogeneous Transformations from frame i to frame i-1 when joint i is at zero position
    # Wi are the axis of rotation of i-th joint
    # Use the numeric approximation:   f'(x) = (f(x+delta) - f(x-delta))/(2*delta)
    #
    # You can do the following steps:
    #     J = matrix of 6x7 full of zeros
    #     q_next = [q1+delta       q2        q3   ....     q7
    #                  q1       q2+delta     q3   ....     q7
    #                              ....
    #                  q1          q2        q3   ....   q7+delta]
    #     q_prev = [q1-delta       q2        q3   ....     q7
    #                  q1       q2-delta     q3   ....     q7
    #                              ....
    #                  q1          q2        q3   ....   q7-delta]
    #     FOR i = 1,..,7:
    #           i-th column of J = ( FK(i-th row of q_next) - FK(i-th row of q_prev) ) / (2*delta_q)
    #     RETURN J
    #
    J = numpy.asarray([[0.0 for a in q] for i in range(6)])
    
    for i in range(len(q)):
        # Create q_next and q_prev by adding and subtracting delta_q to the i-th joint
        q_next = numpy.copy(q)
        q_prev = numpy.copy(q)
        q_next[i] += delta_q
        q_prev[i] -= delta_q
        
        # Calculate forward kinematics for q_next and q_prev
        fk_next = forward_kinematics(q_next, T, W)
        fk_prev = forward_kinematics(q_prev, T, W)
        
        # Compute the i-th column of the Jacobian
        J[:, i] = (fk_next - fk_prev) / (2 * delta_q)
    
    return J

def get_q_distance(q, init_guess):
    # Calculate the distance between two joint configurations
    # q is the current configuration
    # init_guess is the initial configuration
    return numpy.linalg.norm(numpy.array(q) - numpy.array(init_guess))




    

def inverse_kinematics(x, y, z, roll, pitch, yaw, T, W, init_guess=numpy.zeros(7), max_iter=80):
    pd = numpy.asarray([x, y, z, roll, pitch, yaw])
    TOL = 1e-6  # Tolerance for error
    q = init_guess
    iterations = 0

    while iterations < max_iter:
        # Calculate Forward Kinematics 'p'
        p = forward_kinematics(q, T, W)
        
        # Calculate error = p - pd
        error = p - pd
        
        # Ensure orientation angles of error are in [-pi, pi]
        error[3:] = numpy.mod(error[3:] + numpy.pi, 2 * numpy.pi) - numpy.pi
        
        # Imprimir información relevante solo para las primeras 15 iteraciones
        if iterations < 15:
            print(f"[IK][Iter {iterations}] Error vector: {error}")
            print(f"[IK][Iter {iterations}] Error norm: {numpy.linalg.norm(error):.6f}")
            print(f"[IK][Iter {iterations}] q: {q}")

        # Check if the error is within tolerance
        if numpy.linalg.norm(error) < TOL:
            break
        
        # Calculate Jacobian
        J = jacobian(q, T, W)
        
        # Update q estimation with q = q - pseudo_inverse(J) * error
        q = q - numpy.dot(numpy.linalg.pinv(J), error)
        
        # Ensure all angles q are in [-pi, pi]
        q = numpy.mod(q + numpy.pi, 2 * numpy.pi) - numpy.pi
        
        # Increment iterations
        iterations += 1

    # Set success if maximum iterations were not exceeded and calculated angles are in valid range
    success = iterations < max_iter and angles_in_joint_limits(q)

    # Mostrar resumen final
    print(f"[IK] Resultado: éxito={success}, iteraciones={iterations}, distancia={get_q_distance(q, init_guess):.6f}")
    
    # Ruta absoluta del archivo CSV
    file_path = os.path.expanduser("~/results.csv")

    try:
        with open(file_path, "a") as file:
            file.write(f"{success}, {iterations}, {get_q_distance(q, init_guess)}\n")
        print(f"[IK] Resultados guardados en: {file_path}")
    except Exception as e:
        print(f"[IK] Error al guardar CSV: {e}")
    return success, q









   
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
    trj = JointTrajectory()
    trj.header.stamp = rospy.Time.now()
    q = initial_guess
    for i in range(len(X)):
        x, y, z, roll, pitch, yaw = X[i]
        success, q = inverse_kinematics(x, y, z, roll, pitch, yaw, transforms, W, q, max_iterations)
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
        init_guess = req.initial_guess.data
    else:
        init_guess = req.initial_guess
    resp = InverseKinematicsPose2PoseResponse()
    success, q = inverse_kinematics(x, y, z, R, P, Y, init_guess, max_iterations)
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