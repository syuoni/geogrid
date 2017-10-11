# -*- coding: utf-8 -*-
import numpy as np
from Coord2Dist import Coord2Dist as C2D
from PIL import Image
from Heat import Heat

class GridNet(object):
    def __init__(self, n_x, n_y, range_x, range_y):
        self.n_x = int(n_x)
        self.n_y = int(n_y)
        self.n_grid = self.n_x * self.n_y
        self.min_x, self.max_x = range_x
        self.min_y, self.max_y = range_y
        
        self.delta_x = (self.max_x-self.min_x) / self.n_x
        self.delta_y = (self.max_y-self.min_y) / self.n_y
    
    @staticmethod
    def float_xrange(x1, x2):
        x = int(x1)
        while x <= x2:
            if x >= x1:
                yield x
            x += 1
    
    # start from the point 1, i.e. xy1. 
    # there may be several cross points for ONE grid in line, keep the cross point nearest to start point. 
    def cross_points_in_line(self, xy1, xy2, drop_start=True):
        x1, y1 = xy1
        x2, y2 = xy2
        i1, j1 = self.xy2ij(xy1, in_range=False, return_integer=False)
        i2, j2 = self.xy2ij(xy2, in_range=False, return_integer=False)
        
        min_i, max_i = (i1, i2) if i1 < i2 else (i2, i1)
        min_j, max_j = (j1, j2) if j1 < j2 else (j2, j1)
        
        cross_points = {}
        # if start point and end point NOT in a row, then the line may cross same horizontal grid lines
        if min_i < max_i:
            # Iteration for these horizontal lines, OR, rows
            for i in GridNet.float_xrange(min_i, max_i):
                # check if the row index in rational range
                if 0 <= i <= self.n_y:
                    # calculate the (x, y) of the cross point
                    y = self.max_y - self.delta_y*i
                    x = x1 + (x2 - x1) * (y - y1) / (y2 - y1)
                    # check if x in rational range
                    if self.min_x <= x <= self.max_x:
                        dist = C2D.coord2dist_scalar(xy1, (x, y))
                        
                        # check the TWO grids above and below the cross point
                        idx = self.xy2idx((x, y - self.delta_y/2))
                        if idx is not None and (idx not in cross_points or dist < C2D.coord2dist_scalar(xy1, cross_points[idx])):
                            cross_points[idx] = x, y
                        idx = self.xy2idx((x, y + self.delta_y/2))
                        if idx is not None and (idx not in cross_points or dist < C2D.coord2dist_scalar(xy1, cross_points[idx])):
                            cross_points[idx] = x, y
                        
        # if start point and end point NOT in a column, then the line may cross same vertical grid lines
        if min_j < max_j:
            for j in GridNet.float_xrange(min_j, max_j):
                if 0 <= j <= self.n_x:
                    x = self.min_x + self.delta_x*j
                    y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
                    if self.min_y <= y <= self.max_y:
                        dist = C2D.coord2dist_scalar(xy1, (x, y))
                        
                        # check the TWO grids on the left and right of the cross point                        
                        idx = self.xy2idx((x - self.delta_x/2, y))
                        if idx is not None and (idx not in cross_points or dist < C2D.coord2dist_scalar(xy1, cross_points[idx])):
                            cross_points[idx] = x, y
                        idx = self.xy2idx((x + self.delta_x/2, y))
                        if idx is not None and (idx not in cross_points or dist < C2D.coord2dist_scalar(xy1, cross_points[idx])):
                            cross_points[idx] = x, y
        
        idx = self.xy2idx(xy1)
        if idx is not None and idx in cross_points:
            if drop_start:
                del cross_points[idx]
            else:
                # set the cross point with the grid where xy1 is in, is just xy1 (distance is 0)
                cross_points[idx] = xy1
        return cross_points
    
    def check_idx(self, idx):
        return 0 <= idx < self.n_grid

    def check_ij(self, ij):
        i, j = ij
        return (0 <= i < self.n_y) and (0 <= j < self.n_x)
    
    def ij_list2vec(self, ij_list):
        vec = np.zeros(self.n_grid)
        for ij in ij_list:
            if self.check_ij(ij):
                vec[self.ij2idx(ij)] = 1
        return vec
            
    def idx_list2vec(self, idx_list):
        vec = np.zeros(self.n_grid)
        for idx in idx_list:
            if self.check_idx(idx):
                vec[idx] = 1
        return vec
    
    def xy2ij(self, xy, in_range=True, return_integer=True):
        x, y = xy
        i = (self.max_y - y) / self.delta_y
        j = (x - self.min_x) / self.delta_x
        if in_range and not self.check_ij((i, j)):
            return None
        if return_integer:
            i, j = int(i), int(j)
        return i, j
    
    def xy2idx(self, xy):
        ij = self.xy2ij(xy, in_range=True, return_integer=True)
        if ij is None:
            return None
        else:
            return self.ij2idx(ij)
            
    def ij2center_xy(self, ij):
        if self.check_ij(ij):
            i, j = ij
            cx = self.min_x + self.delta_x * (j + 0.5)
            cy = self.max_y - self.delta_y * (i + 0.5)
            return cx, cy
        else:
            return None
        
    def idx2center_xy(self, idx):
        ij = self.idx2ij(idx)
        if ij is None:
            return None
        else:
            return self.ij2center_xy(ij)
            
    def ij2idx(self, ij):
        if self.check_ij(ij):
            i, j = ij
            return i*self.n_x + j
        else:
            return None

    def idx2ij(self, idx):
        if self.check_idx(idx):
            i = idx // self.n_x
            j = idx % self.n_x
            return i, j
        else:
            return None
    
    def xy2center_xy(self, xy):
        ij = self.xy2ij(xy, in_range=True, return_integer=True)
        if ij is None:
            return None
        else:
            return self.ij2center_xy(ij)
    
    def vec2img(self, vec, scale=1, use_color=True):
        matrix = vec.reshape((self.n_y, self.n_x))
        img = Matrix2Img.matrix2img(matrix, scale=scale, use_color=use_color)
        return img
    
    def path2vec(self, path):
        vec = np.zeros(self.n_grid)
        
        for i in range(len(path)-1):
            xy1 = path[i]
            xy2 = path[i+1]
            cross_points = self.cross_points_in_line(xy1, xy2, drop_start=False)
            vec += self.idx_list2vec(cross_points.keys())
        return np.where(vec>1, 1, vec)
    
    def img2vec(self, img, scale=1, positive_rgb=None):
        if positive_rgb is None:
            # default positive color (where vec[idx] is 1) is white
            pr, pg, pb = 255, 255, 255
        else:
            pr, pg, pb = positive_rgb
        
        vec = np.zeros(self.n_grid)
        for i in range(self.n_y):
            for j in range(self.n_x):
                r, g, b = img.getpixel((j*scale, i*scale))
                if r==pr and g==pg and b==pb:
                    idx = self.ij2idx((i, j))
                    vec[idx] = 1
        return vec
    
class Matrix2Img(object):
    @staticmethod
    def matrix2img(matrix, scale=1, use_color=True):
        n_y, n_x = matrix.shape
        min_v, max_v = np.min(matrix), np.max(matrix)
        if min_v == max_v:
            min_v -= 0.5
            max_v += 0.5
        matrix = (matrix - min_v) / (max_v - min_v)
        
        if use_color:
            color_values = Heat.get_color_values(matrix).transpose((1, 2, 0))
        else:
            color_values = Heat.get_grey_values(matrix).transpose((1, 2, 0))
        
        img = Image.fromarray(color_values)
        img = img.resize((n_x*scale, n_y*scale))
        return img
    
    @staticmethod
    def add_background(img, bg_img):
        img_data = np.array(img)
        bg_data = np.array(bg_img.resize(img.size))
        assert img_data.shape[-1] == 3 and bg_data.shape[-1] == 4
        
        # show img where backgroud is transparent entirely
        img_data = np.where(bg_data[:, :, 3:4]==0, img_data, bg_data[:, :, 0:3])
        return Image.fromarray(img_data)
    
    
if __name__ == '__main__':
    GN = GridNet(400, 400, (121.24, 121.66), (31.02, 31.38))
    print(GN.idx2center_xy(1))
    print(GN.xy2idx((121.240001, 31.020001)))
    
    cross_points = GN.cross_points_in_line((121.24001, 31.02), (121.66, 31.38))
    vec = GN.idx_list2vec(cross_points.keys())
    img = GN.vec2img(vec, scale=2, use_color=False)
    img.show()
    