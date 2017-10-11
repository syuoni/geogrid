# -*- coding: utf-8 -*-
import numpy as np

class Coord2Dist(object):
    '''transformation of coordinates and distance'''
    # kelometer as unit
    R = 6378.137
    
    @staticmethod
    def coord2dist_scalar(xy1, xy2, center_y=None):
        x1, y1 = xy1
        x2, y2 = xy2
        dist_matrix = Coord2Dist.coord2dist_matrix([x1], [y1], [x2], [y2], center_y=center_y)
        return dist_matrix[0, 0]
    
    @staticmethod
    def coord2dist_matrix(x1_vec, y1_vec, x2_vec=None, y2_vec=None, center_y=None):
        # x1 and y1 as rows, x2 and y2 as columns
        x1_vec = np.array(x1_vec)
        y1_vec = np.array(y1_vec)
        if x2_vec == None or y2_vec == None:
            x2_vec = x1_vec
            y2_vec = y1_vec
        else:
            x2_vec = np.array(x2_vec)
            y2_vec = np.array(y2_vec)
        
        if center_y is None:
            center_y = (y1_vec.mean() + y2_vec.mean()) / 2
        
        x_ratio = np.pi * Coord2Dist.R / 180 * np.cos(center_y * np.pi / 180)
        y_ratio = np.pi * Coord2Dist.R / 180 
        
        # -1 indicate the unspecified dimension
        dist_x = (x1_vec.reshape((-1, 1)) - x2_vec) * x_ratio
        dist_y = (y1_vec.reshape((-1, 1)) - y2_vec) * y_ratio
        return (dist_x**2 + dist_y**2) ** 0.5
    
    @staticmethod
    def dist2dx(dist, center_y):
        x_ratio = np.pi * Coord2Dist.R / 180 * np.cos(center_y * np.pi / 180)
        dx = dist / x_ratio
        return dx
    
    @staticmethod
    def dist2dy(dist):
        y_ratio = np.pi * Coord2Dist.R / 180 
        dy = dist / y_ratio
        return dy
        
    
if __name__ == '__main__':
    x1 = [117.22082, 117.22798, 117, 118]
    y1 = [31.843278, 31.850011, 31,  31]
    x2 = [118]
    y2 = [31]    
    
    print(Coord2Dist.coord2dist_matrix(x1, y1))
    print(Coord2Dist.coord2dist_matrix(x1, y1, x2, y2))
    print(Coord2Dist.coord2dist_scalar((118, 31), (117, 31)))
    
    print(Coord2Dist.dist2dx(2, center_y=60))
    print(Coord2Dist.dist2dy(2))
