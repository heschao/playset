import os

import cv2
import numpy as np
from card_img import getCards

from playset.attributes import Card
from playset.attributes import Purple, Green, Red
from playset.identify import CardRepo, get_features, id_attributes, get_card_contours, extract_card, center

root = 'C:/Users/Elmacho/Downloads/documents-export-2016-04-11'
filename = '20160411_210523.jpg'
orig = cv2.imread(os.path.join(root, filename))
img = cv2.resize(orig,(0,0),None,fx=0.5,fy=0.5)
card_contours = [x for x in get_card_contours(img, 16)]
cp = img.copy()
# cv2.drawContours(cp, card_contours, -1, (0, 255, 0), 3)

n = len(card_contours)
attr = [None] * n
icp = img.copy()
for i in range(0, n):
    contour = card_contours[i]
    card = extract_card(contour, img)
    if card is None:
        continue
    try:
        a = id_attributes(card) # type: Card
        attr[i] = a
        x,y,w,h = cv2.boundingRect(contour)

        font = cv2.FONT_HERSHEY_SIMPLEX
        size = 0.75
        inc = 25
        x0 = x+50
        y0 = y+50
        cv2.putText(icp,str(a.count),(x0,y0), font, size,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(icp,str(a.color.name),(x0,y0+inc), font, size,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(icp,str(a.shading),(x0,y0+inc*2), font, size,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(icp,str(a.shape),(x0,y0+inc*3), font, size,(0,0,0),2,cv2.LINE_AA)
        cv2.drawContours(icp,card_contours,i,(0,255,0),3)
    except:
        pass
cv2.imshow('image',icp)


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


