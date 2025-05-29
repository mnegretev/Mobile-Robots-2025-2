#!/usr/bin/env python3
#
# MOBILE ROBOTS - FI-UNAM, 2025-2
# INVERSE KINEMATICS USING NEWTON-RAPHSON

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
    x, y, z = H[0, 3], H[1, 3], H[2, 3]
    R, P, Y = list(tft.euler_from_matrix(H))
    return numpy.asarray([x, y, z, R, P, Y])

def jacobian(q, T, W):
    delta_q = 1e-6
    J = numpy.zeros((6, len(q)))
    qn = numpy.asarray([q] * len(q)) + delta_q * numpy.identity(len(q))
    qp = numpy.asarray([q] * len(q)) - delta_q * numpy.identity(len(q))
    for i in range(len(q)):
        J[:, i] = (forward_kinematics(qn[i], T, W) - forward_kinematics(qp[i], T, W)) / (2.0 * delta_q)
    return J

def inverse_kinematics(x, y, z, roll, pitch, yaw, T, W, init_guess=numpy.zeros(7), max_iter=100):
    pd = numpy.asarray([x, y, z, roll, pitch, yaw])
    q = init_guess.copy()
    p = forward_kinematics(q, T, W)
    iterations = 0
    error = p - pd
    error[3:6] = (error[3:6] + math.pi) % (2*math.pi) - math.pi
    tolerance = 1e-4
    λ = 0.01  # Factor de amortiguación

    while numpy.linalg.norm(error) > tolerance and iterations < max_iter:
        J = jacobian(q, T, W)
        JT = J.T
        JJt = J @ JT
        damped_inv = JT @ numpy.linalg.inv(JJt + (λ**2) * numpy.identity(6))
        dq = -damped_inv @ error
        q = (q + dq + math.pi) % (2*math.pi) - math.pi  # Mantener ángulos en [-π, π]
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
    req.time_step = time_step
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
        found = False
        for joint in robot_model.joints:
            if joint.name == name:
                if joint.axis is None:
                    rospy.logerr(f"[Model] Joint {name} has no axis defined!")
                joints.append(joint)
                found = True
                break
        if not found:
            rospy.logerr(f"[Model] Joint {name} not found in URDF!")
    for joint in joints:
        T = tft.translation_matrix(joint.origin.xyz)
        R = tft.euler_matrix(*joint.origin.rpy)
        transforms.append(tft.concatenate_matrices(T, R))
    if len(joints) != 7 or len(transforms) != 7:
        rospy.logwarn(f"[Model] Warning: Expected 7 joints, got {len(joints)}")
    transforms.append(tft.identity_matrix())  # dummy final link
    return joints, transforms

def angles_in_joint_limits(q):
    for i in range(len(q)):
        if q[i] < joints[i].limit.lower or q[i] > joints[i].limit.upper:
            rospy.logwarn(f"{prompt}Joint {i} value {q[i]:.2f} out of bounds ({joints[i].limit.lower:.2f}, {joints[i].limit.upper:.2f})")
            return False
    return True

def callback_forward_kinematics(req):
    resp = ForwardKinematicsResponse()
    if len(req.q) != 7:
        rospy.logwarn(f"{prompt}Expected 7 DOF, got {len(req.q)}.")
        resp.x = resp.y = resp.z = 0.0
        resp.roll = resp.pitch = resp.yaw = 0.0
        return resp
    W = [j.axis for j in joints]
    resp.x, resp.y, resp.z, resp.roll, resp.pitch, resp.yaw = forward_kinematics(req.q, transforms, W)
    return resp

def get_trajectory_time(p1, p2, speed_factor):
    m = max(numpy.abs(numpy.asarray(p1) - numpy.asarray(p2)))
    return m / speed_factor + 0.5

def callback_ik_for_trajectory(req):
    global max_iterations
    rospy.loginfo(f"{prompt}IK trajectory requested to: {[req.x, req.y, req.z, req.roll, req.pitch, req.yaw]}")
    initial_guess = req.initial_guess or rospy.wait_for_message("/hardware/arm/current_pose", Float64MultiArray).data
    W = [j.axis for j in joints]
    p1 = forward_kinematics(initial_guess, transforms, W)
    p2 = [req.x, req.y, req.z, req.roll, req.pitch, req.yaw]
    t = req.duration if req.duration > 0 else get_trajectory_time(p1, p2, 0.25)
    dt = req.time_step if req.time_step > 0 else 0.05

    try:
        X, T = get_polynomial_trajectory_multi_dof(p1, p2, duration=t, time_step=dt)
    except Exception as e:
        rospy.logerr("Error in polynomial trajectory: %s", str(e))
        return InverseKinematicsPose2TrajResponse(articular_trajectory=JointTrajectory())

    if len(X) == 0:
        rospy.logwarn("Generated polynomial trajectory is empty.")
        return InverseKinematicsPose2TrajResponse(articular_trajectory=JointTrajectory())

    trj = JointTrajectory()
    trj.header.stamp = rospy.Time.now()
    q = initial_guess

    for i, pose in enumerate(X):
        success, q = inverse_kinematics(*pose, transforms, W, q, max_iterations)
        if not success:
            rospy.logwarn(f"IK failed at point {i}")
            return InverseKinematicsPose2TrajResponse(articular_trajectory=JointTrajectory())
        p = JointTrajectoryPoint()
        p.positions = q
        p.time_from_start = rospy.Duration.from_sec(T[i])
        trj.points.append(p)

    return InverseKinematicsPose2TrajResponse(articular_trajectory=trj)

def callback_ik_for_pose(req):
    global max_iterations
    rospy.loginfo(f"{prompt}IK pose requested: {[req.x, req.y, req.z, req.roll, req.pitch, req.yaw]}")
    initial_guess = req.initial_guess or rospy.wait_for_message("/hardware/arm/current_pose", Float64MultiArray).data
    W = [j.axis for j in joints]
    success, q = inverse_kinematics(req.x, req.y, req.z, req.roll, req.pitch, req.yaw, transforms, W, initial_guess, max_iterations)
    resp = InverseKinematicsPose2PoseResponse()
    resp.q = q if success else []
    if not success:
        rospy.logwarn(f"{prompt}IK failed for pose.")
    return resp

def main():
    global joint_names, max_iterations, joints, transforms, prompt
    rospy.init_node("ik_geometric")
    prompt = rospy.get_name().upper() + ".->"
    joint_names = rospy.get_param("~joint_names", [])
    max_iterations = rospy.get_param("~max_iterations", 100)
    rospy.loginfo(f"{prompt}Joint names: {joint_names}")
    rospy.loginfo(f"{prompt}Max iterations: {max_iterations}")

    joints, transforms = get_model_info(joint_names)
    if len(joints) < 7 or len(transforms) < 8:
        rospy.logerr("URDF parsing failed. Not enough joints or transforms.")
        sys.exit(-1)

    rospy.Service("/manipulation/forward_kinematics", ForwardKinematics, callback_forward_kinematics)
    rospy.Service("/manipulation/ik_trajectory", InverseKinematicsPose2Traj, callback_ik_for_trajectory)
    rospy.Service("/manipulation/ik_pose", InverseKinematicsPose2Pose, callback_ik_for_pose)
    rospy.loginfo(f"{prompt}IK node initialized and services ready.")
    rospy.spin()

if __name__ == "__main__":
    main()
