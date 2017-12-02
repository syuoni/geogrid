# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import shapefile
from shapely.geometry import Point, shape
from itertools import cycle
from Coord2Dist import Coord2Dist

class ShapeMap(object):
    def __init__(self, shp_fn, key):
        self.sf = shapefile.Reader(shp_fn)
        self.shapes = self.sf.shapes()
        self.sf_recs = self.sf.records()
        self.sf_df = pd.DataFrame(self.sf_recs, columns=[field_name for field_name, *_ in self.sf.fields[1:]])
        
        # key is the identifier of polygons
        if isinstance(key, str):
            self.key = key
        elif isinstance(key, list):
            self.key = '_'.join(key)
            self.sf_df[self.key] = self.sf_df[key[0]]
            for k in key[1:]:
                self.sf_df[self.key] = self.sf_df[self.key] + '-' + self.sf_df[k]
        
        self.geo_shapes = []
        self.shapely_shapes = []
        shapely_data = []
        for idx in self.sf_df.index:
            geo_shape = self.shapes[idx].__geo_interface__
            shapely_shape = shape(geo_shape)
            self.geo_shapes.append(geo_shape)
            self.shapely_shapes.append(shapely_shape)
            shapely_data.append((shapely_shape.centroid.x, shapely_shape.centroid.y, shapely_shape.area, shapely_shape.length))
        shapely_data = pd.DataFrame(shapely_data, columns=['center_x', 'center_y', 'shapely_area', 'shapely_length'])
        self.sf_df = pd.concat([self.sf_df, shapely_data], axis=1)
        
    def coord2key(self, x, y, max_tries=None):
        p = Point(x, y)        
        dist = Coord2Dist.coord2dist_matrix(self.sf_df['center_x'].values, self.sf_df['center_y'].values, [x], [y]).flatten()
        
        idx_search_seq = np.argsort(dist)
        if max_tries is not None:
            idx_search_seq = idx_search_seq[:max_tries]
        for idx in idx_search_seq:
            if self.shapely_shapes[idx].contains(p):
                key_res = self.sf_df.loc[idx, self.key]
                break
        else:
            key_res = None
        return key_res
       
    def visualize(self, ax=None, color_col=None, cm=None):
        if color_col is None:
            poly_dic = {'default': []}
            cmap = {'default': ['w', 'k']}
        else:
            assert color_col in self.sf_df.columns
            
            color_values = [cv for cv in self.sf_df[color_col].unique() if pd.notnull(cv)]
            poly_dic = {poly_class: [] for poly_class in color_values}
            poly_dic['N/A'] = []
            # Only treat float as numerical, otherwise category
            if all([isinstance(cv, float) for cv in color_values]):
                cm = plt.cm.rainbow if cm is None else cm
                
                min_v, max_v = min(color_values), max(color_values)
                cmap = {poly_class: [cm((poly_class-min_v)/(max_v-min_v)), 'k'] for poly_class in color_values}
                cmap['N/A'] = ['w', 'gray']
            else:
                cm = plt.cm.Set1 if cm is None else cm
                
                cmap = {poly_class: [c, 'k'] for poly_class, c in zip(color_values, cycle(cm.colors))}
                cmap['N/A'] = ['w', 'gray']
        
        if ax is None:
            fig, ax = plt.subplots()
        for idx in self.sf_df.index:
            geo_shape = self.geo_shapes[idx]
            
            poly_class = self.sf_df.at[idx, color_col] if color_col is not None else 'default'
            poly_class = 'N/A' if pd.isnull(poly_class) else poly_class
            if geo_shape['type'] == 'Polygon':
                poly_dic[poly_class].append(Polygon(np.array(geo_shape['coordinates'][0])))
            elif geo_shape['type'] == 'MultiPolygon':
                poly_dic[poly_class].extend([Polygon(np.array(coords[0])) for coords in geo_shape['coordinates']])
            else:
                print(idx, geo_shape['type'])
        
        for poly_class in poly_dic.keys():
            ax.add_collection(PatchCollection(poly_dic[poly_class], linewidth=0.5, 
                                              facecolor=cmap[poly_class][0], edgecolor=cmap[poly_class][1]))
        ax.axis('scaled')
        return ax


if __name__ == '__main__':
    sm_cn = ShapeMap(shp_fn='CHN_adm_shp/CHN_adm2', key=['NAME_1', 'NAME_2'])
    # Jiaxing
    print(sm_cn.coord2key(120.960474, 30.789884))
    # Tianjin
    print(sm_cn.coord2key(117.172931, 39.146498))
    # None
    print(sm_cn.coord2key(135.717406, 34.626706))
    sm_cn.visualize(color_col='NAME_1')
    
    sm_nyc = ShapeMap(shp_fn='nyct2010_16b/nyct2010.shp', key='CT2010')
    # 024800
    print(sm_nyc.coord2key(914408.0, 124617.0))
    # None 
    print(sm_nyc.coord2key(0.0, 0.0))
    sm_nyc.visualize(color_col='BoroCode')
    
    sm_nyc_zip = ShapeMap(shp_fn='ZIP_CODE_040114/ZIP_CODE_040114.shp', key='ZIPCODE')
    sm_nyc_zip.visualize(color_col='POPULATION')
    
    