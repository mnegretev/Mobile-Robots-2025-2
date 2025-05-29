#!/usr/bin/env python3
#
# MOBILE ROBOTS - FI-UNAM, 2025-2
# COLOR SEGMENTATION USING HSV (REVISED)
#

import numpy as np
import cv2
import ros_numpy
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PointStamped, Point
from vision_msgs.srv import RecognizeObject, RecognizeObjectResponse

NAME = "Lujan Pérez Carlos Eduardo"

def segment_by_color(img_bgr, points, obj_name):
    global img_hsv, img_bin, img_filtered
    img_x, img_y, x, y, z = 0, 0, 0, 0, 0

    # Convertir imagen BGR a HSV
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Definir máscara de color para cada objeto
    if obj_name == "pringles":
        img_bin = cv2.inRange(img_hsv, (25, 150, 50), (35, 255, 255))  # Amarillo
    elif obj_name == "drink":
        img_bin = cv2.inRange(img_hsv, (15, 200, 50), (25, 255, 255))  # Naranja
    else:
        rospy.logwarn("Objeto desconocido: " + obj_name)
        return [0, 0, 0, 0, 0]

    # Filtrado morfológico para eliminar ruido
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    img_filtered = cv2.erode(img_bin, kernel)
    img_filtered = cv2.dilate(img_filtered, kernel)

    # Calcular centroid usando momentos
    moments = cv2.moments(img_filtered, binaryImage=True)
    if moments["m00"] != 0:
        img_x = int(moments["m10"] / moments["m00"])
        img_y = int(moments["m01"] / moments["m00"])
    else:
        rospy.logwarn("No se pudo calcular el centroide para: " + obj_name)
        return [0, 0, 0, 0, 0]

    # Obtener píxeles segmentados
    nonz = cv2.findNonZero(img_filtered)
    if nonz is None or len(nonz) == 0:
        rospy.logwarn("No se segmentaron píxeles para: " + obj_name)
        return [img_x, img_y, 0, 0, 0]

    # Calcular promedio de coordenadas 3D válidas
    valid_points = 0
    for [[c, r]] in nonz:  # c = columna (x), r = fila (y)
        pt = points[r, c]
        if not np.isfinite(pt[0]) or not np.isfinite(pt[1]) or not np.isfinite(pt[2]):
            continue
        # Opcional: descartar puntos fuera de rango razonable
        if pt[2] < 0.3 or pt[2] > 2.5:
            continue
        x += pt[0]
        y += pt[1]
        z += pt[2]
        valid_points += 1

    if valid_points == 0:
        rospy.logwarn("Todos los puntos 3D son inválidos para: " + obj_name)
        return [img_x, img_y, 0, 0, 0]

    # Promedio
    x /= valid_points
    y /= valid_points
    z /= valid_points

    return [img_x, img_y, x, y, z]

def callback_find_object(req):
    global pub_point, img_bgr
    rospy.loginfo("Buscando objeto: " + req.name)

    # Convertir PointCloud2 a imagen BGR
    arr = ros_numpy.point_cloud2.pointcloud2_to_array(req.point_cloud)
    rgb_arr = arr['rgb'].copy()
    rgb_arr.dtype = np.uint32
    r = np.asarray(((rgb_arr >> 16) & 255), dtype='uint8')
    g = np.asarray(((rgb_arr >> 8) & 255), dtype='uint8')
    b = np.asarray(((rgb_arr) & 255), dtype='uint8')
    img_bgr = cv2.merge((b, g, r))

    # Ejecutar segmentación
    [img_x, img_y, x, y, z] = segment_by_color(img_bgr, arr, req.name)

    # Publicar resultado
    resp = RecognizeObjectResponse()
    resp.recog_object.header.frame_id = 'kinect_link'
    resp.recog_object.header.stamp = rospy.Time.now()
    resp.recog_object.pose.position.x = x
    resp.recog_object.pose.position.y = y
    resp.recog_object.pose.position.z = z

    pub_point.publish(PointStamped(
        header=resp.recog_object.header,
        point=Point(x=x, y=y, z=z)
    ))

    # Dibujar círculo en la imagen BGR (c = columna, r = fila)
    if img_x != 0 and img_y != 0:
        cv2.circle(img_bgr, (int(img_x), int(img_y)), 20, [0, 255, 0], thickness=3)

    return resp

def main():
    global pub_point, img_bgr, img_hsv, img_bin, img_filtered
    rospy.init_node("color_segmentation")
    rospy.Service("/vision/obj_reco/detect_and_recognize_object", RecognizeObject, callback_find_object)
    pub_point = rospy.Publisher('/detected_object', PointStamped, queue_size=10)

    # Inicialización de imágenes para visualización
    img_bgr = np.zeros((480, 640, 3), np.uint8)
    img_hsv = np.zeros((480, 640, 3), np.uint8)
    img_bin = np.zeros((480, 640), np.uint8)       # Binaria 1 canal
    img_filtered = np.zeros((480, 640), np.uint8)  # Filtrada 1 canal

    rospy.loginfo("Segmentación de color iniciada - " + NAME)
    loop = rospy.Rate(10)
    while not rospy.is_shutdown():
        cv2.imshow("BGR", img_bgr)
        cv2.imshow("HSV", img_hsv)
        cv2.imshow("Binary Mask", img_bin)
        cv2.imshow("Filtered Mask", img_filtered)
        cv2.waitKey(1)
        loop.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

