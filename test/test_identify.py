import os
from os.path import basename, dirname

import cv2

from playset.identify import get_features


def test_identify_33():
    path = dirname(os.path.abspath(__file__))
    root = os.path.join(path,'..','data','training')
    filename = os.path.join(root,'33.png')
    img = cv2.imread(filename)
    features = get_features(img,215,500)
    cv2.drawContours(img, features, -1, (0, 0, 0), 2)
    cv2.imshow('image', img)
    cv2.waitKey()
    assert(len(features)==3)


def test_identify_6():
    path = dirname(os.path.abspath(__file__))
    root = os.path.join(path,'..','data','training')
    filename = os.path.join(root,'6.png')
    img = cv2.imread(filename)
    features = get_features(img)
    assert(len(features)==3)


def test_identify_1():
    path = dirname(os.path.abspath(__file__))
    root = os.path.join(path,'..','data','training')
    filename = os.path.join(root,'0.png')
    img = cv2.imread(filename)
    features = get_features(img)
    assert(len(features)==1)