# -*- coding: utf-8 -*-
import numpy as np
from scipy.interpolate import interp1d
from PIL import Image, ImageDraw, ImageFont

class Heat(object):
    '''Heat color map'''
    _red   = interp1d(x=[0.00, 0.35, 0.66, 0.89, 1.00], 
                      y=[0,    0,    255,  255,  128 ])    
    _green = interp1d(x=[0.00, 0.12, 0.37, 0.64, 0.91, 1.00],
                      y=[0,    0,    255,  255,  0,    0   ])    
    _blue  = interp1d(x=[0.00, 0.11, 0.34, 0.65, 1.00],
                      y=[128,  255,  255,  0,    0   ])
    _grey  = interp1d(x=[0.00, 1.00],
                      y=[0,    255])
    
    @staticmethod   
    def get_grey_values(ratio_values):
        assert isinstance(ratio_values, np.ndarray) and ((ratio_values>=0)&(ratio_values<=1)).all()
        
        grey_values = Heat._grey(ratio_values)
        grey_values = np.round(grey_values).astype(np.uint8)
        return np.array([grey_values, grey_values, grey_values])
    
    @staticmethod
    def get_grey(ratio_value):
        return tuple([x[0] for x in Heat.get_grey_values(np.array([ratio_value]))])
    
    @staticmethod
    def get_color_values(ratio_values):
        assert isinstance(ratio_values, np.ndarray) and ((ratio_values>=0)&(ratio_values<=1)).all()
        
        red_values   = Heat._red(ratio_values)
        red_values   = np.round(red_values).astype(np.uint8)
        green_values = Heat._green(ratio_values)
        green_values = np.round(green_values).astype(np.uint8)
        blue_values  = Heat._blue(ratio_values)
        blue_values  = np.round(blue_values).astype(np.uint8)        
        return np.array([red_values, green_values, blue_values])
    
    @staticmethod
    def get_color(ratio_value):
        return tuple([x[0] for x in Heat.get_color_values(np.array([ratio_value]))])    

    @staticmethod
    def color_bar(value_range, h=500, w=20, font_size=15, w_multiple=3, n_divide=5):
        minv, maxv = value_range
        img = Image.new('RGB', (w*w_multiple, h), (255, 255, 255))
        
        draw = ImageDraw.Draw(img)
        for i in range(h):
            ratio_value = i / float(h)
            rgb = Heat.get_color(ratio_value)
            draw.rectangle((0, i, w-1, i+1), fill=rgb)
            
        draw.line((w, 0, w, h-1), fill=(0, 0, 0))
        
        font = ImageFont.truetype('arial.ttf', font_size)
        for k in range(n_divide+1):
            ratio = float(k) / n_divide
            if ratio < 1:
                draw.text((w+2, ratio*h), '%.1f' % (ratio*maxv + (1-ratio)*minv), fill=(0, 0, 0), font=font)
            else:
                draw.text((w+2, ratio*h-font_size), '%.1f' % maxv, fill=(0, 0, 0), font=font)
            draw.line((0.5*w, ratio*h, 1.5*w, ratio*h), fill=(0, 0, 0))
            
        return img
        

if __name__ == '__main__':
    print(Heat.get_color(0))
    print(Heat.get_color(0.00001))
    print(Heat.get_color(0.7))
    print(Heat.get_color(1))
    print(Heat.get_color(0.99999))
    # print(Heat.get_color(2))
    print(Heat.get_grey(1))
    print(Heat.get_grey(0))
    print(Heat.get_grey(0.5))
    print(Heat.get_grey(0.9999))
    
    value_range = (20, 93)
    img = Heat.color_bar(value_range, font_size=20, w_multiple=5, n_divide=10)
    img.show()
    