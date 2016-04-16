import logging
import os

import cv2
import sys

from playset.identify import set_from_img

root = 'C:/Users/Elmacho/Downloads/documents-export-2016-04-11'
# name = '20160411_210957.jpg'
# full = os.path.join(root, name)
# orig = cv2.imread(full)
# img = cv2.resize(orig, (0, 0), None, fx=0.5, fy=0.5)
# set_from_img(img, name)
# sys.exit(0)

for name in os.listdir(root):
    full = os.path.join(root,name)
    if not os.path.isfile(full):
        continue
    orig = cv2.imread(full)
    img = cv2.resize(orig,(0,0),None,fx=0.5,fy=0.5)
    try:
        set_from_img(img,name)
    except Exception, e:
        logging.error('failed for {:} with {:}'.format(name,e))
sys.exit(0)


# root = 'c:/users/elmacho/pycharmprojects/playset/data/training'
# repo = CardRepo(root)
#
# card = repo.get(1)
# cv2.imshow('image',card)
# attributes = id_attributes(card.copy())
# print(attributes.color)
# print(attributes.count)
# print(attributes.shading)
# print(attributes.shape)




# h = cv2.calcHist([hsvcard], [0, 1], None, [180, 256], [0, 180, 0, 256])
# x = np.array(h)
# thresh = 100
# i,j = np.where(x>=thresh)
# print(i)
# print(j)


