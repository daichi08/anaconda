#! /usr/bin/env python
# -*- coding: utf-8 -*-

import math

AREA_SIZE = 10

class precastDB:
    def __init__(self, x_p, y_p, dist, angle, x_i, y_i):
        self.x_p = x_p
        self.y_p = y_p
        self.dist = dist
        self.angle = angle
        self.x_i = x_i
        self.y_i = y_i

def calc_grid_map(obs_x, obs_y, grid_size):
    x_min = round(min(obs_x) - AREA_SIZE/2.0)
    y_min = round(min(obs_y) - AREA_SIZE/2.0)
    x_max = round(max(obs_x) + AREA_SIZE/2.0)
    y_max = round(max(obs_y) + AREA_SIZE/2.0)
    x_range = int(round((x_max - x_min)/grid_size))
    y_range = int(round((y_max - y_min)/grid_size))
    
    return x_min, y_min, x_max, y_max, x_range, y_range

def atan_zero_to_twopi(x, y):
    angle = math.atan2(y, x)
    if angle < 0.0:
        angle += 2*math.pi
        
    return angle

def precasting(x_min, y_min, x_range, y_range, grid_size, phi_res):
    precast = [[] for i in range(int(round(2*math.pi/phi_res))+1)]
    
    for x_i in range(x_range):
        for y_i in range(y_range):
            x_p = x_i * grid_size + x_min
            y_p = y_i * grid_size + y_min
            
            dist = math.sqrt(x_p**2 + y_p**2)
            angle = atan_zero_to_twopi(x_p, y_p)
            angle_id = int(math.floor(angle/phi_res))
            
            pcd = precastDB(x_p, y_p, dist, angle, x_i, y_i)
            
            precast[angle_id].append(pcd)

    return precast

def ray_casting(obs_x, obs_y, grid_size, phi_res):
    x_min, y_min, x_max, y_max, x_range, y_range = calc_grid_map(obs_x, obs_y, grid_size)
    precast = precasting(x_min, y_min, x_range, y_range, grid_size, phi_res)
    grid_map = [[0.0 for i in range(y_range)] for i in range(x_range)]
    
    for (x, y) in zip(obs_x, obs_y):
        dist = math.sqrt(x**2 + y**2)
        angle = atan_zero_to_twopi(x, y)
        angle_id = int(math.floor(angle/phi_res))
        
        grid_list = precast[angle_id]
        
        x_i = int(round((x-x_min)/grid_size))
        y_i = int(round((y-y_min)/grid_size))
        
        for grid in grid_list:
            if grid.dist > dist:
                grid_map[grid.x_i][grid.y_i] += 0.5
        grid_map[x_i][y_i] += 1
    return grid_map, x_min, y_min

def main(obstacles=[]):
    grid_size = 0.2 #[m]
    phi_res   = 3 * math.pi/180
    
    for i in range(1): # while rosok
        obs_x = [obs[0] for obs in obstacles]
        obs_y = [obs[1] for obs in obstacles]
        grid_map, x_min, y_min = ray_casting(obs_x, obs_y, grid_size, phi_res)
    return grid_map, x_min, y_min, grid_size

if __name__ == "__main__":
    print("start")
    obstacles = [[8-i/10,2+i/10] for i in range(61)]
    grid_map, x_min, y_min, grid_size = main(obstacles)
    
    import csv
    with open("gridmap.csv", "w") as file:
        write = csv.writer(file, lineterminator="\n")
        write.writerows(grid_map)