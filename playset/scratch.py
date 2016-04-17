import logging
import os

import cv2
import sys

from playset.identify import set_from_img, get_card_contours, extract_card, get_features

root = 'C:/Users/Elmacho/Downloads/documents-export-2016-04-11'
name = '20160411_210633.jpg'
full = os.path.join(root, name)
orig = cv2.imread(full)
img = cv2.resize(orig, (0, 0), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
cv2.imshow('image',img)
card_contours = [x for x in get_card_contours(img, 20)]

i = -1
i += 1
print('i={:}'.format(i))
contour = card_contours[i]
card = extract_card(contour, img)
features = get_features(card)
cv2.imshow('image',card)
gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
cmask = np.zeros(gray.shape)
enc_center, enc_radius = cv2.minEnclosingCircle(features[0])
center = (int(enc_center[0]), int(enc_center[1]))
# rr  = 7
# radius = int(enc_radius/rr)
# cv2.circle(cmask, center, radius, (1), -1)
(x0, y0) = center
w = 100
h = 30
tl = (x0 - w / 2, y0 - h / 2)
lr = (x0 + w / 2, y0 + h / 2)
cv2.rectangle(cmask, tl, lr, (1), -1)
x = gray.copy()
x[cmask == 0] = 0
feature_intensity = float(x.sum().sum()) / float(cmask.sum().sum())

y = gray.copy()
fmask = np.zeros(gray.shape)
cv2.drawContours(fmask, features, -1, (1), -1)
bmask = 1 - fmask
y[bmask == 0] = 0
bg_intensity = float(y.sum().sum()) / float(bmask.sum().sum())

points = [0, 1, 5, 10, 50, 90, 95, 99, 100]
xprctiles = np.percentile(x[x > 0], points)
yprctiles = np.percentile(y[y > 0], points)
print(xprctiles)
print(yprctiles)
print('avg feature intensity / 5%: {:.2f}'.format(feature_intensity / xprctiles[2]))
print('5% feature / background: {:.2f}'.format(xprctiles[2]/yprctiles[2]))
print [feature_intensity / bg_intensity, feature_intensity, bg_intensity]

blur_sigma = 1000
threshold_val = 200
blur = cv2.GaussianBlur(x, (1, 1), blur_sigma)
flag, thresh = cv2.threshold(blur, threshold_val, 255, cv2.THRESH_BINARY)
cv2.imshow('image',thresh)
lines = cv2.HoughLines(thresh,50,0.1,1)
print(len(lines))

set_from_img(img, name)
sys.exit(0)

for name in os.listdir(root):
    full = os.path.join(root,name)
    if not os.path.isfile(full):
        continue
    orig = cv2.imread(full)
    img = cv2.resize(orig,(0,0),None,fx=0.3,fy=0.3)
    try:
        set_from_img(img,'image')
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


