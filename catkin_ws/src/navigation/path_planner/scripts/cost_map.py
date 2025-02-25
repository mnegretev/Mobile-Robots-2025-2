#!/usr/bin/env python3
#
# MOBILE ROBOTS - FI-UNAM, 2025-2
# COST MAPS
#
# Instructions:
# Write the code necesary to get a cost map given
# an occupancy grid map and a cost radius.
#

import rospy
import numpy
from nav_msgs.msg import OccupancyGrid
from nav_msgs.srv import GetMap
from nav_msgs.srv import GetMapResponse
from nav_msgs.srv import GetMapRequest

NAME = "Carlos Castañeda Mora"

def get_cost_map(static_map, cost_radius):
    cost_map = numpy.copy(static_map)
    [height, width] = static_map.shape
    #
    # TODO:
    # Write the code necessary to calculate a cost map for the given map.
    # To calculate cost, consider as example the following map:    
    # [[ 0 0 0 0 0 0]
    #  [ 0 X 0 0 0 0]
    #  [ 0 X X 0 0 0]
    #  [ 0 X X 0 0 0]
    #  [ 0 X 0 0 0 0]
    #  [ 0 0 0 X 0 0]]
    # Where occupied cells 'X' have a value of 100 and free cells have a value of 0.
    # Cost is an integer indicating how near cells and obstacles are:
    # [[ 3 3 3 2 2 1]
    #  [ 3 X 3 3 2 1]
    #  [ 3 X X 3 2 1]
    #  [ 3 X X 3 2 2]
    #  [ 3 X 3 3 3 2]
    #  [ 3 3 3 X 3 2]]
    # Cost_radius indicate the number of cells around obstacles with costs greater than zero.
    for i in range(height):
        for j in range(width):
            if static_map[i,j] > 50:
                for k1 in range(-cost_radius, cost_radius+1):
                    for k2 in range(-cost_radius, cost_radius+1):
                        cost = cost_radius-max(abs(k1),abs(k2)) +1
                        cost_map[i+k1,j+k2] = max(cost, cost_map[i+k1, j+k2])
    return cost_map

def callback_cost_map(req):
    global cost_map
    return GetMapResponse(map=cost_map)
    
def main():
    global cost_map, inflated_map
    print("COST MAPS - " + NAME)
    rospy.init_node("cost_map")
    rospy.wait_for_service('/static_map')
    grid_map = rospy.ServiceProxy("/static_map", GetMap)().map
    map_info = grid_map.info
    width, height, res = map_info.width, map_info.height, map_info.resolution
    grid_map = numpy.reshape(numpy.asarray(grid_map.data, dtype='int'), (height, width))
    rospy.Service('/cost_map'    , GetMap, callback_cost_map)
    loop = rospy.Rate(1)
    
    cost_radius = rospy.get_param("~cost_radius", 0.1)
    while not rospy.is_shutdown():
        if cost_radius > 1.0:
            cost_radius = 1.0
        print("Calculating cost map with " +str(round(cost_radius/res)) + " cells")
        cost_map_data = get_cost_map(grid_map, round(cost_radius/res))
        cost_map_data = numpy.ravel(numpy.reshape(cost_map_data, (width*height, 1)))
        cost_map      = OccupancyGrid(info=map_info, data=cost_map_data) 
        loop.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    
