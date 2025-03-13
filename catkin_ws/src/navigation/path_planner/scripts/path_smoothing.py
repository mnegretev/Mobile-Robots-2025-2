#!/usr/bin/env python3

import numpy as np
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import Pose, PoseStamped, Point
from navig_msgs.srv import ProcessPath, ProcessPathResponse

NAME = "CARLOS EDUARDO LUJAN PEREZ"

def smooth_path(Q, alpha, beta, max_steps):
    """
    Aplica descenso de gradiente para suavizar la ruta.

    :param Q: Ruta original (lista de puntos 2D)
    :param alpha: Peso del término de suavidad
    :param beta: Peso del término de adherencia a la ruta original
    :param max_steps: Número máximo de iteraciones
    :return: Ruta suavizada P
    """
    steps = 0
    tol = 1e-5
    epsilon = 0.1  # Factor de ajuste del descenso de gradiente

    P = np.copy(Q).astype(float)  
    nabla = np.zeros_like(Q)  # Gradiente inicializado en 0

    print(f"Ruta inicial:\n{P}")  # Verificar la ruta antes del suavizado

    while np.linalg.norm(nabla) > tol and steps < max_steps:
        prev_P = np.copy(P)  # Guardamos la versión anterior para comparar cambios

        for i in range(1, len(Q) - 1):  # No modificamos los extremos
            nabla[i] = alpha * (2 * P[i] - P[i - 1] - P[i + 1]) + beta * (P[i] - Q[i])
        
        # Verificar si realmente se están realizando cambios
        if np.linalg.norm(nabla) < tol:
            print(f"Convergencia alcanzada en {steps} iteraciones.")
            break

        P -= nabla * epsilon  # Aplicamos descenso de gradiente
        steps += 1  

        # Verificar si hubo cambio
        if np.allclose(prev_P, P, atol=1e-6):
            print(f"Pequeños cambios en la ruta, terminando en {steps} iteraciones.")
            break

    print(f"Ruta suavizada:\n{P}")  # Verificar la ruta después del suavizado
    return P

def callback_smooth_path(req):
    global msg_smooth_path

    # Obtener parámetros desde ROS
    alpha = rospy.get_param('~alpha', 0.1)
    beta = rospy.get_param('~beta', 0.9)
    steps = rospy.get_param('~steps', 1000)

    print(f"Smoothing path with params: alpha={alpha}, beta={beta}, steps={steps}")
    
    start_time = rospy.Time.now()

    # Convertir la ruta del mensaje a un array numpy
    Q = np.array([[p.pose.position.x, p.pose.position.y] for p in req.path.poses])
    
    if Q.shape[0] < 3:
        print("Error: La ruta debe tener al menos 3 puntos para ser suavizada.")
        return ProcessPathResponse(processed_path=msg_smooth_path)

    # Aplicar suavizado
    P = smooth_path(Q, alpha, beta, steps)

    end_time = rospy.Time.now()
    print(f"Path smoothed in {1000 * (end_time - start_time).to_sec()} ms")

    # Convertir el resultado a mensaje Path
    msg_smooth_path.poses = [
        PoseStamped(pose=Pose(position=Point(x=p[0], y=p[1]))) for p in P
    ]

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
