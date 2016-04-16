import os

import cv2
import numpy as np

from playset.attributes import Purple, Green, Red
from playset.identify import CardRepo, get_features, id_attributes

root = 'c:/users/elmacho/pycharmprojects/playset/data/training'
repo = CardRepo(root)

card = repo.get(1)
cv2.imshow('image',card)
attributes = id_attributes(card.copy())
print(attributes.color)
print(attributes.count)
print(attributes.shading)
print(attributes.shape)




# h = cv2.calcHist([hsvcard], [0, 1], None, [180, 256], [0, 180, 0, 256])
# x = np.array(h)
# thresh = 100
# i,j = np.where(x>=thresh)
# print(i)
# print(j)


