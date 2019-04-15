#! /usr/bin/env python
# -*- coding: utf-8 -*-

import math
from scipy.stats import norm

AREA_SIZE = 10
GRID_SIZE = 0.2
ANGLE_RES = 3 * math.pi/180

class precastDB:
    def __init__(self, x_p, y_p, dist, angle, x_i, y_i):
        self.x_p = x_p
        self.y_p = y_p
        self.dist = dist
        self.angle = angle
        self.x_i = x_i
        self.y_i = y_i

def atan_zero_to_twopi(x, y):
    angle = math.atan2(y, x)
    if angle < 0.0:
        angle += 2*math.pi
        
    return angle

def precasting(x_min, y_min, x_range, y_range):
    precast = [[] for i in range(int(round(2*math.pi/ANGLE_RES))+1)]
    
    for x_i in range(x_range):
        for y_i in range(y_range):
            x_p = x_i * GRID_SIZE + x_min
            y_p = y_i * GRID_SIZE + y_min
            
            dist = math.sqrt(x_p**2 + y_p**2)
            angle = atan_zero_to_twopi(x_p, y_p)
            angle_id = int(math.floor(angle/ANGLE_RES))
            
            pcd = precastDB(x_p, y_p, dist, angle, x_i, y_i)
            
            precast[angle_id].append(pcd)
    return precast

def calc_grid_map(obstacles):
    obs_x = [obs[0] for obs in obstacles]
    obs_y = [obs[1] for obs in obstacles]

    x_min = round(min(obs_x) - AREA_SIZE/2.0)
    y_min = round(min(obs_y) - AREA_SIZE/2.0)
    x_max = round(max(obs_x) + AREA_SIZE/2.0)
    y_max = round(max(obs_y) + AREA_SIZE/2.0)
    x_range = int(round((x_max - x_min)/GRID_SIZE))
    y_range = int(round((y_max - y_min)/GRID_SIZE))
    
    return x_min, y_min, x_range, y_range

def ray_casting(obstacles):
    x_min, y_min, x_range, y_range = calc_grid_map(obstacles)
    precast = precasting(x_min, y_min, x_range, y_range)
    grid_map = [[0.0 for i in range(y_range)] for i in range(x_range)]
    obstacles_ids = []

    for (x, y) in obstacles:        
        dist = math.sqrt(x**2 + y**2)
        angle = atan_zero_to_twopi(x, y)
        angle_id = int(math.floor(angle/ANGLE_RES))
        
        grid_list = precast[angle_id]
        
        x_i = int(round((x-x_min)/GRID_SIZE))
        y_i = int(round((y-y_min)/GRID_SIZE))
        obstacles_ids.append([x_i, y_i])
        
        for grid in grid_list:
            if grid.dist > dist:
                if grid_map[grid.x_i][grid.y_i] < 1:
                    grid_map[grid.x_i][grid.y_i] = 0.5
        grid_map[x_i][y_i] = 1
    return x_min, y_min, x_range, y_range, grid_map, obstacles_ids

def main(paths, obstacles):
    x_min, y_min, x_range, y_range, grid_map, obstacles_ids = ray_casting(obstacles)
    
    unique_obs_ids = set(list(map(tuple, obstacles_ids)))
    
    paths_ids = [
        [[int(round((x-x_min)/GRID_SIZE)), int(round((y-y_min)/GRID_SIZE))] for (x, y, th, v, o) in path]
        for path in paths
    ]
    unique_paths_ids = set(list(map(tuple, sum(paths_ids, []))))
    
    for (path_xid, path_yid) in unique_paths_ids:
        min_dist = float("inf")
        for (obs_xid, obs_yid) in unique_obs_ids:
            dist = math.sqrt((path_xid-obs_xid)**2 + (path_yid-obs_yid)**2)
            if min_dist > dist:
                min_dist = dist
        grid_score = 1.0 - norm.cdf(min_dist, 0.0, 3.0)
        grid_map[path_xid][path_yid] += grid_score

    return grid_map
    


if __name__ == "__main__":
    paths = [[1, 1, math.pi/2, 0, 1],
             [0, 1, math.pi/2, 1, 1],
             [1.5, 2, math.pi/2, 0,1]]
    obstacles = [[3,0],[10, 10]]
    grid_map = main(paths, obstacles)
    
    import csv
    with open("gridmap.csv", "w") as file:
        write = csv.writer(file, lineterminator="\n")
        write.writerows(grid_map)