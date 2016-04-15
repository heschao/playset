from collections import namedtuple

import cv2
import os

import sys
import matplotlib.pyplot as plt
import numpy as np
from card_img import getCards
import card_img
import pandas as pd


Purple = 'purple'
Green = 'green'
Red = 'red'
Solid = 'solid'
Shaded = 'shaded'
Empty = 'empty'
Squiggle = 'squiggle'
Diamond = 'diamond'
Oval = 'oval'
Card = namedtuple('Card','color count shading shape')
truth = {
    0:Card(Purple,3,Solid,Squiggle),
    1:Card(Green,2,Solid,Diamond),
    2:Card(Purple,2,Empty,Diamond),
    3:Card(Green,3,Shaded,Squiggle),
    4:Card(Red,2,Shaded,Diamond),
    5:Card(Green,3,Empty,Oval),
    6:Card(Red,1,Solid,Squiggle),
    8:Card(Green,2,Empty,Oval)
}

# a = cv2.goodFeaturesToTrack(image=(1),maxCorners=5,qualityLevel=0.1,minDistance=100)

c = card(1)
cv2.imshow('image',c)
hsvcard = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
h = cv2.calcHist([hsvcard], [0, 1], None, [180, 256], [0, 180, 0, 256])
x = np.array(h)
thresh = 1e3
i,j = np.where(x>=thresh)
lower = np.array([70,250,0])
upper = np.array([80,256,256])
b = cv2.inRange(hsvcard,lower,upper)



def card(n):
    return cv2.imread(os.path.join(ROOT,'{:}.png'.format(n)))

def compare_color(im1,im2):
    hlim = 180
    slim = 256
    h1 = cv2.calcHist([cv2.cvtColor(im1,cv2.COLOR_BGR2HSV)], [0, 1], None, [hlim,slim], [0, hlim, 0, slim])
    h2 = cv2.calcHist([cv2.cvtColor(im2,cv2.COLOR_BGR2HSV)], [0, 1], None, [hlim,slim], [0, hlim, 0, slim])
    return cv2.compareHist(h1,h2,cv2.HISTCMP_CORREL)


def hist_hsv(card,hlim,slim):
    hsvcard = cv2.cvtColor(card,cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsvcard], [0, 1], None, [hlim,slim], [0, hlim, 0, slim])
    plt.clf()
    plt.imshow(h, interpolation='nearest')
    return h


def hist_rgb(card):
    chans = cv2.split(card)
    colors = ("b", "g", "r")
    plt.clf()
    plt.title("'Flattened' Color Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    features = []

    # loop over the image channels
    for (chan, color) in zip(chans, colors):
        # create a histogram for the current channel and
        # concatenate the resulting histograms for each
        # channel
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)

        # plot the histogram
        plt.plot(hist, color=color)
        plt.xlim([0, 256])


def extract_cards(filename, n):
    assert(os.path.isfile(filename))
    img = cv2.imread(filename)
    return [x for x in getCards(img, n)]


def get_features(img,threshold_val=None,blur_sigma=1000):
    """
    Get features of the right size, and with right spacing, to avoid double contours for rings
    :param img:
    :return:
    """
    if not threshold_val:
        for threshold_val in [150,200]:
            a = get_features(img, threshold_val)
            if len(a) in [1, 2, 3]:
                return a


    minarea = 16000
    maxarea = 30000
    mindist = 50

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (1, 1), blur_sigma)
    flag, thresh = cv2.threshold(blur, threshold_val, 255, cv2.THRESH_BINARY)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    a = [cv2.contourArea(x) for x in contours]
    y = []
    z = []
    for i in range(0,len(a)):
        if (a[i] <= maxarea)&(a[i]>= minarea):
            y.append(contours[i])
            z.append(a[i])

    dmat = np.zeros((len(y),len(y)))
    for i in range(0,len(y)):
        for j in range(i+1,len(y)):
            c = y[i]
            d = y[j]
            cx,cy = center(c)
            dx,dy = center(d)
            dmat[i,j] = np.sqrt((dx - cx) ** 2 + (dy - cy) ** 2)
    np.set_printoptions(precision=1)

    t = []
    for u in np.argwhere(dmat < mindist) :
        if u[0] >= u[1]:
            continue
        if z[u[0]] > z[u[1]]:
            t.append(u[1])
        else:
            t.append(u[0])

    print t
    w = []
    for i in range(0,len(y)):
        if i not in t:
            w.append(y[i])
    return  w


def center(cnt):
    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return cx,cy


def extract_and_save(filename,ncards,dirout,offset):
    cards = extract_cards(filename,ncards)
    print '{:}/{:} found'.format(len(cards),ncards)
    for i in range(0, ncards):
        filenumber = i + offset
        reffile = os.path.join(ROOT, 'data', 'training', '{:}.png'.format(filenumber))
        cv2.imwrite(reffile, cards[i])


def guess_shape(features):
    a = np.mean(np.array([cv2.contourArea(x) for x in features]))
    if a > 25e3:
        return 'oval'
    elif a < 20e3:
        return 'diamond'
    else:
        return 'squiggle'


if __name__ == "__main__":
    ROOT = 'c:/users/elmacho/pycharmprojects/playset/data/training'
    # reffile = 'c:/users/elmacho/downloads/all cards.jpg'
    # testfile = 'c:/users/elmacho/downloads/set sample.jpg'
    #
    # offset=30
    # filename = 'C:/Users/Elmacho/Downloads/documents-export-2016-04-11/20160411_211151.jpg'
    # ncards = 9
    # extract_and_save(filename,ncards,os.path.join(ROOT,'data','training'),offset)
    #
    # sys.exit(0)


    for filename in os.listdir(ROOT):
        card = cv2.imread(os.path.join(ROOT,filename))
        features = get_features(card)
        shp = guess_shape(features)
        print('{:}: {:} {:}{:}'.format(filename,len(features),shp, 's' if len(features)>1 else ''))
        cv2.drawContours(card, features, -1, (0, 0, 0), 2)
        cv2.imshow('image', card)
        if cv2.waitKey() == 113:
            print('quit')
            sys.exit(0)
        else:
            print('continue')




    sys.exit(0)
    #
    #
    # cv2.drawContours(img, big, 0, (0,255,0), 3)
    # cv2.imshow('image',img)
    # cv2.waitKey()
    #
    # card = big[0]
    # peri = cv2.arcLength(card,True)
    # approx = cv2.approxPolyDP(card,0.02*peri,True)
    #
    # rect = cv2.minAreaRect(card)
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    # cv2.drawContours(img,[box],0,(0,0,255),2)
    #
    # h = np.array([ [0,0],[449,0],[449,449],[0,449] ],np.float32)
    # a = np.array([ x[0] for x in approx ],np.float32)
    # transform = cv2.getPerspectiveTransform(a,h)
    # warp = cv2.warpPerspective(img,transform,(450,450))
    # cv2.imshow('image',warp)
    # cv2.waitKey()
    #
    # sys.exit(0)
    #
    #
    #
    #
    # sys.exit(0)
    #
    #
    # # shwo color image
    # img = cv2.imread(reffile)
    # img2 = img[:,:,::-1]
    # plt.imshow(img2)
    # plt.show()
    # sys.exit(0)
    #
    #
