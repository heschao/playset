from collections import namedtuple
import numpy as np

HsvColor = namedtuple('HsvColor', 'name lower upper')

Purple = HsvColor(name='purple',
                  lower=np.array([130, 175, 0]),
                  upper=np.array([160, 256, 256]))
Green = HsvColor(name='green',
                 lower=np.array([38, 180, 0]),
                 upper=np.array([80, 256, 256]))
Red = HsvColor(name='red',
               lower=np.array([160, 180, 0]),
               upper=np.array([179, 256, 256]))
Orange = HsvColor(name='orange',
                  lower=np.array([0, 180, 0]),
               upper=np.array([8, 256, 256]))

Solid = 'solid'
Shaded = 'shaded'
Empty = 'empty'
Squiggle = 'squiggle'
Diamond = 'diamond'
Oval = 'oval'
Card = namedtuple('Card','color count shading shape')

AttributeKey = {
    'count' : {1:0,2:1,3:2},
    'color': {'red':0,'green':1,'purple':2},
    'shading': {Empty:0,Shaded:1,Solid:2},
    'shape':{Diamond:0,Oval:1,Squiggle:2}
}