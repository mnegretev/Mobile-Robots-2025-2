#!/usr/bin/env python3
#
# MOBILE ROBOTS - FI-UNAM, 2025-2
# INVERSE KINEMATICS USING NEWTON-RAPHSON
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
NAME = "Lujan Perez Carlos Eduardo"
   
def forward_kinematics(q, T, W):
    H = tft.identity_matrix()
    for i in range(len(q)):
        H = tft.concatenate_matrices(H, T[i], tft.rotation_matrix(q[i], W[i]))
    H = tft.concatenate_matrices(H, T[7])
    x, y, z = H[0,3], H[1,3], H[2,3]
    R, P, Y = list(tft.euler_from_matrix(H)) 
    return numpy.asarray([x,y,z,R,P,Y])

def jacobian(q, T, W):
    delta_q = 1e-6
    J = numpy.zeros((6, len(q)))
    qn = numpy.asarray([q,]*len(q)) + delta_q * numpy.identity(len(q))
    qp = numpy.asarray([q,]*len(q)) - delta_q * numpy.identity(len(q))
    for i in range(len(q)):
        J[:, i] = (forward_kinematics(qn[i], T, W) - forward_kinematics(qp[i], T, W)) / (2.0 * delta_q)
    return J

def inverse_kinematics(x, y, z, roll, pitch, yaw, T, W, init_guess=numpy.zeros(7), max_iter=100):
    pd = numpy.asarray([x,y,z,roll,pitch,yaw])
    q = init_guess
    p = forward_kinematics(q, T, W)
    iterations = 0
    error = p - pd
    error[3:6] = (error[3:6] + math.pi) % (2*math.pi) - math.pi
    tolerance = 1e-4
    while numpy.linalg.norm(error) > tolerance and iterations < max_iter:
        J = jacobian(q, T, W)
        J_inv = numpy.linalg.pinv(J)
        q = (q - numpy.dot(J_inv, error) + math.pi)% (2*math.pi) - math.pi
        p = forward_kinematics(q, T, W)
        error = p - pd
        error[3:6] = (error[3:6] + math.pi) % (2*math.pi) - math.pi
        iterations += 1
    success = numpy.linalg.norm(error) < tolerance
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
##########################################################################################################
##########################################################################################################
##########################################################################################################
def callback_ik_for_trajectory(req):
    global max_iterations
    Pd = [req.x, req.y, req.z, req.roll, req.pitch, req.yaw]
    print(prompt + "Calculating IK and trajectory for " + str(Pd))

    if not req.initial_guess:
        initial_guess = rospy.wait_for_message("/hardware/arm/current_pose", Float64MultiArray)
        initial_guess = initial_guess.data
    else:
        initial_guess = req.initial_guess

    W = [joints[i].axis for i in range(len(joints))]  
    p1 = forward_kinematics(initial_guess, transforms, W)
    p2 = [req.x, req.y, req.z, req.roll, req.pitch, req.yaw]
    t  = req.duration if req.duration > 0 else get_trajectory_time(p1, p2, 0.25)
    dt = req.time_step if req.time_step > 0 else 0.05

    try:
        X, T = get_polynomial_trajectory_multi_dof(p1, p2, duration=t, time_step=dt)
    except Exception as e:
        rospy.logerr("Error in polynomial trajectory service: %s", str(e))
        resp = InverseKinematicsPose2TrajResponse()
        resp.articular_trajectory = JointTrajectory()
        return resp

    trj = JointTrajectory()
    trj.header.stamp = rospy.Time.now()
    q = initial_guess

    for i in range(len(X)):
        x, y, z, roll, pitch, yaw = X[i]
        success, q = inverse_kinematics(x, y, z, roll, pitch, yaw, transforms, W, q, max_iterations)
        if not success:
            rospy.logwarn("IK failed at trajectory point %d", i)
            resp = InverseKinematicsPose2TrajResponse()
            resp.articular_trajectory = JointTrajectory()
            return resp

        p = JointTrajectoryPoint()
        p.positions = q
        p.time_from_start = rospy.Duration.from_sec(T[i])
        trj.points.append(p)

    resp = InverseKinematicsPose2TrajResponse()
    resp.articular_trajectory = trj
    return resp
   ###############################################################################################################################################################
####################################################################################################################################################################################################################
##########################################################################################################
def callback_ik_for_pose(req):
    global max_iterations
    [x,y,z,R,P,Y] = [req.x,req.y,req.z,req.roll,req.pitch,req.yaw]
    print(prompt+"Calculating inverse kinematics for pose: " + str([x,y,z,R,P,Y]))
    if len(req.initial_guess) <= 0 or req.initial_guess == None:
        init_guess = rospy.wait_for_message("/hardware/arm/current_pose", Float64MultiArray, 5.0)
        init_guess = init_guess.data
    else:
        init_guess = req.initial_guess
    resp = InverseKinematicsPose2PoseResponse()
    success, q = inverse_kinematics(x, y, z, R, P, Y, transforms, [j.axis for j in joints], init_guess, max_iterations)
   if not success:
      rospy.logwarn("IK failed for pose.")
      resp = InverseKinematicsPose2PoseResponse()
      resp.q = []  # trayectoria vacía
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
    loop = rospy.Rate(40)
    while not rospy.is_shutdown():
        loop.sleep()

if __name__ == '__main__':
    main()

