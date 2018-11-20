# -*- coding: utf-8 -*-
import numpy as np

class Coord2Dist(object):
    '''transformation of coordinates and distance'''
    # Kelometer as unit
    R = 6371.004 
    
    @staticmethod
    def coord2dist_scalar(xy1, xy2, center_y=None):
        x1, y1 = xy1
        x2, y2 = xy2
        dist_matrix = Coord2Dist.coord2dist_matrix([x1], [y1], [x2], [y2], center_y=center_y)
        return dist_matrix[0, 0]
    
    
    @staticmethod
    def coord2dist_matrix_sphere(x1_vec, y1_vec, x2_vec=None, y2_vec=None):
        '''Distance on the surface of sphere. (20181120 updated)
        
        Distance between (LonA, LatA) and (LonB, LatB) 
        C = sin(LatA)*sin(LatB) + cos(LatA)*cos(LatB)*cos(LonA-LonB)
        Distance = R*Arccos(C)*Pi/180
        '''
        # x1 and y1 as rows, x2 and y2 as columns
        x1_vec = np.asarray(x1_vec) * np.pi / 180
        y1_vec = np.asarray(y1_vec) * np.pi / 180
        if x2_vec is None or y2_vec is None:
            x2_vec = x1_vec
            y2_vec = y1_vec
        else:
            x2_vec = np.array(x2_vec) * np.pi / 180
            y2_vec = np.array(y2_vec) * np.pi / 180
        
        C = np.sin(y1_vec[:, None]) * np.sin(y2_vec) + np.cos(y1_vec[:, None]) * np.cos(y2_vec) * np.cos(x1_vec[:, None] - x2_vec)
        C = np.clip(C, -1, 1)
        return Coord2Dist.R * np.arccos(C)
    
    
    @staticmethod
    def coord2dist_matrix_plane(x1_vec, y1_vec, x2_vec=None, y2_vec=None):
        '''Distance on the projected plane. 
        '''
        # x1 and y1 as rows, x2 and y2 as columns
        x1_vec = np.asarray(x1_vec)
        y1_vec = np.asarray(y1_vec)
        if x2_vec is None or y2_vec is None:
            x2_vec = x1_vec
            y2_vec = y1_vec
        else:
            x2_vec = np.array(x2_vec)
            y2_vec = np.array(y2_vec)
        
        dist_x = (x1_vec[:, None] - x2_vec) * Coord2Dist.R * np.pi / 180 * np.cos((y1_vec[:, None] + y2_vec) / 2 * np.pi / 180)
        dist_y = (y1_vec[:, None] - y2_vec) * Coord2Dist.R * np.pi / 180 
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
    # Beijing, Shanghai, Guangzhou, Shenzhen, Chengdu
    x1 = [116.405338, 121.48284, 113.276028, 114.062827, 104.07923]
    y1 = [39.916237,  31.234941, 23.136399 , 22.552194,  30.655823]
    
    dm_plane = Coord2Dist.coord2dist_matrix_plane(x1, y1)
    dm_sphere = Coord2Dist.coord2dist_matrix_sphere(x1, y1)
    print(dm_sphere / dm_plane)
    
#    x2 = [118]
#    y2 = [31]
    
#    print(Coord2Dist.coord2dist_matrix(x1, y1, x2, y2))
#    print(Coord2Dist.coord2dist_scalar((118, 31), (117, 31)))
#    
#    print(Coord2Dist.dist2dx(2, center_y=60))
#    print(Coord2Dist.dist2dy(2))
