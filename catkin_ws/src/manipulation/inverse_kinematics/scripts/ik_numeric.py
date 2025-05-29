#!/usr/bin/env python3
#
# MOBILE ROBOTS - FI-UNAM, 2025-2
# INVERSE KINEMATICS USING NEWTON-RAPHSON
# Autor: Lujan Perez Carlos Eduardo

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

# Variables globales
prompt = ""
NAME = "Lujan Perez Carlos Eduardo"

# ----------------------------------------------------------------------------------------
# FUNCIONES PRINCIPALES DE CINEMÁTICA
# ----------------------------------------------------------------------------------------

# Cinemática directa: calcula pose del efector final dada una configuración articular q
def forward_kinematics(q, T, W):
    H = tft.identity_matrix()
    for i in range(len(q)):
        H = tft.concatenate_matrices(H, T[i], tft.rotation_matrix(q[i], W[i]))
    H = tft.concatenate_matrices(H, T[7])
    
    x, y, z = H[0, 3], H[1, 3], H[2, 3]
    R, P, Y = list(tft.euler_from_matrix(H))
    return numpy.asarray([x, y, z, R, P, Y])

# Cálculo del Jacobiano por diferencias finitas
def jacobian(q, T, W):
    delta_q = 1e-6
    J = numpy.zeros((6, len(q)))
    qn = numpy.asarray([q] * len(q)) + delta_q * numpy.identity(len(q))
    qp = numpy.asarray([q] * len(q)) - delta_q * numpy.identity(len(q))

    for i in range(len(q)):
        J[:, i] = (forward_kinematics(qn[i], T, W) - forward_kinematics(qp[i], T, W)) / (2.0 * delta_q)

    return J

# Cinemática inversa con Newton-Raphson y pseudoinversa con amortiguamiento
def inverse_kinematics(x, y, z, roll, pitch, yaw, T, W, init_guess=numpy.zeros(7), max_iter=250):
    pd = numpy.asarray([x, y, z, roll, pitch, yaw])
    q = init_guess.copy()
    p = forward_kinematics(q, T, W)
    iterations = 0

    error = p - pd
    error[3:6] = (error[3:6] + math.pi) % (2 * math.pi) - math.pi

    tolerance = 1e-4
    λ = 0.01  # Parámetro de amortiguamiento

    while numpy.linalg.norm(error) > tolerance and iterations < max_iter:
        J = jacobian(q, T, W)
        JT = J.T
        JJt = J @ JT
        damped_inv = JT @ numpy.linalg.inv(JJt + (λ**2) * numpy.identity(6))

        dq = -damped_inv @ error
        q = (q + dq + math.pi) % (2 * math.pi) - math.pi

        p = forward_kinematics(q, T, W)
        error = p - pd
        error[3:6] = (error[3:6] + math.pi) % (2 * math.pi) - math.pi

        iterations += 1

    success = numpy.linalg.norm(error) < tolerance
    return success, q

# ----------------------------------------------------------------------------------------
# UTILIDADES
# ----------------------------------------------------------------------------------------

# Obtiene trayectoria polinomial interpolada en espacio cartesiano
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

# Extrae información del modelo URDF
def get_model_info(joint_names):
    robot_model = urdf_parser_py.urdf.URDF.from_parameter_server()
    joints = []
    transforms = []

    for name in joint_names:
        for joint in robot_model.joints:
            if joint.name == name:
                joints.append(joint)
                break

    for joint in joints:
        T = tft.translation_matrix(joint.origin.xyz)
        R = tft.euler_matrix(*joint.origin.rpy)
        transforms.append(tft.concatenate_matrices(T, R))

    transforms.append(tft.identity_matrix())  # Transformación final dummy
    return joints, transforms

# ----------------------------------------------------------------------------------------
# CALLBACKS DE SERVICIOS
# ----------------------------------------------------------------------------------------

# Servicio: Cinemática directa
def callback_forward_kinematics(req):
    resp = ForwardKinematicsResponse()
    if len(req.q) != 7:
        rospy.logwarn(f"{prompt}Expected 7 DOF, got {len(req.q)}.")
        return resp

    W = [numpy.array(j.axis) for j in joints]
    resp.x, resp.y, resp.z, resp.roll, resp.pitch, resp.yaw = forward_kinematics(req.q, transforms, W)
    return resp

# Estimación de duración de trayectoria basada en desplazamiento
def get_trajectory_time(p1, p2, speed_factor):
    m = max(numpy.abs(numpy.asarray(p1) - numpy.asarray(p2)))
    return m / speed_factor + 0.5

# Servicio: IK para trayectoria cartesiana
def callback_ik_for_trajectory(req):
    global max_iterations

    initial_guess = req.initial_guess if len(req.initial_guess) == 7 else rospy.wait_for_message("/hardware/arm/current_pose", Float64MultiArray).data

    W = [numpy.array(j.axis) for j in joints]
    p1 = forward_kinematics(initial_guess, transforms, W)
    p2 = [req.x, req.y, req.z, req.roll, req.pitch, req.yaw]

    t = req.duration if req.duration > 0 else get_trajectory_time(p1, p2, 0.25)
    dt = req.time_step if req.time_step > 0 else 0.05

    try:
        X, T = get_polynomial_trajectory_multi_dof(p1, p2, t, dt)
    except Exception as e:
        rospy.logerr("Trajectory service error: %s", e)
        return InverseKinematicsPose2TrajResponse(articular_trajectory=JointTrajectory())

    trj = JointTrajectory()
    trj.header.stamp = rospy.Time.now()
    q = initial_guess

    for i, pose in enumerate(X):
        success, q = inverse_kinematics(*pose, transforms, W, q, max_iterations)
        if not success:
            rospy.logwarn(f"{prompt}IK failed at step {i}")
            return InverseKinematicsPose2TrajResponse(articular_trajectory=JointTrajectory())

        pt = JointTrajectoryPoint()
        pt.positions = q
        pt.time_from_start = rospy.Duration.from_sec(T[i])
        trj.points.append(pt)

    return InverseKinematicsPose2TrajResponse(articular_trajectory=trj)

# Servicio: IK para una sola pose
def callback_ik_for_pose(req):
    global max_iterations

    initial_guess = req.initial_guess if len(req.initial_guess) == 7 else rospy.wait_for_message("/hardware/arm/current_pose", Float64MultiArray).data
    W = [numpy.array(j.axis) for j in joints]

    success, q = inverse_kinematics(req.x, req.y, req.z, req.roll, req.pitch, req.yaw, transforms, W, initial_guess, max_iterations)

    resp = InverseKinematicsPose2PoseResponse()
    resp.q = q if success else []

    if not success:
        rospy.logwarn(f"{prompt}IK for pose failed.")

    return resp

# ----------------------------------------------------------------------------------------
# INICIALIZACIÓN DEL NODO
# ----------------------------------------------------------------------------------------

def main():
    global joint_names, max_iterations, joints, transforms, prompt

    rospy.init_node("ik_geometric")
    prompt = rospy.get_name().upper() + ".->"

    joint_names = rospy.get_param("~joint_names", [])
    max_iterations = rospy.get_param("~max_iterations", 100)

    joints, transforms = get_model_info(joint_names)
    if len(joints) != 7 or len(transforms) != 8:
        rospy.logerr("URDF error: expected 7 joints and 8 transforms.")
        sys.exit(-1)

    # Advertencia: Asegúrate de remapear estos servicios si tienes múltiples brazos
    rospy.Service("/manipulation/forward_kinematics", ForwardKinematics, callback_forward_kinematics)
    rospy.Service("/manipulation/ik_trajectory", InverseKinematicsPose2Traj, callback_ik_for_trajectory)
    rospy.Service("/manipulation/ik_pose", InverseKinematicsPose2Pose, callback_ik_for_pose)

    rospy.loginfo(f"{prompt}Inverse kinematics service ready.")
    rospy.spin()

if __name__ == "__main__":
    main()

