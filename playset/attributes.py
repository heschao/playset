from collections import namedtuple
import numpy as np

HsvColor = namedtuple('HsvColor', 'name lower upper')

Purple = HsvColor(name='Purple',
                  lower=np.array([130, 175, 0]),
                  upper=np.array([160, 256, 256]))
Green = HsvColor(name='Green',
                 lower=np.array([38, 180, 0]),
                 upper=np.array([80, 256, 256]))
Red = HsvColor(name='Red',
               lower=np.array([160, 180, 0]),
               upper=np.array([179, 256, 256]))
Orange = HsvColor(name='Orange',
                  lower=np.array([0, 180, 0]),
               upper=np.array([8, 256, 256]))

Solid = 'solid'
Shaded = 'shaded'
Empty = 'empty'
Squiggle = 'squiggle'
Diamond = 'diamond'
Oval = 'oval'
Card = namedtuple('Card','color count shading shape')