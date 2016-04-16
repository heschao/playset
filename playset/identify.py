import logging
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from card_img import getCards

# a = cv2.goodFeaturesToTrack(image=(1),maxCorners=5,qualityLevel=0.1,minDistance=100)
#
# c = card(1)
# cv2.imshow('image',c)
# hsvcard = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
# h = cv2.calcHist([hsvcard], [0, 1], None, [180, 256], [0, 180, 0, 256])
# x = np.array(h)
# thresh = 1e3
# i,j = np.where(x>=thresh)
# lower = np.array([70,250,0])
# upper = np.array([80,256,256])
# b = cv2.inRange(hsvcard,lower,upper)

from playset.attributes import Oval, Diamond, Squiggle, Card, Red, Solid, Purple, Green, Orange, Shaded, Empty


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
        return Oval
    elif a < 20e3:
        return Diamond
    else:
        return Squiggle


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


def guess_color(card, features):
    mask = np.zeros(card.shape[0:2], np.uint8)
    cv2.drawContours(mask, features, -1, (255), -1)
    card[mask == 0] = 0
    hsvcard = cv2.cvtColor(card, cv2.COLOR_BGR2HSV)
    colors = [Purple,Green,Red,Orange]
    r = np.zeros((len(colors),1))
    for i in [0,1,2,3]:
        value = colors[i]
        b = cv2.inRange(hsvcard, value.lower, value.upper)
        r[i] = float((b == 255).sum().sum()) / float((mask == 255).sum().sum())
    argmax = np.argmax(r)
    logging.debug('max ratio {:.2f}'.format(r[argmax,0]))
    if argmax==3:
        argmax=2
    result = colors[argmax]
    return result


def guess_shading(card,features):
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    cmask = np.zeros(gray.shape)
    enc_center,enc_radius = cv2.minEnclosingCircle(features[0])
    center = (int(enc_center[0]),int(enc_center[1]))
    radius = int(enc_radius/10)
    cv2.circle(cmask, center, radius, (1), -1)
    x = gray.copy()
    x[cmask == 0] = 0
    feature_intensity = float(x.sum().sum()) / float(cmask.sum().sum())

    y = gray.copy()
    fmask = np.zeros(gray.shape)
    cv2.drawContours(fmask,features,-1,(1),-1)
    bmask = 1-fmask
    y[bmask==0]=0
    bg_intensity = float(y.sum().sum())/float(bmask.sum().sum())

    points = [0, 1, 5, 10, 50, 90, 95, 99, 100]
    xprctiles = np.percentile(x[x > 0], points)
    yprctiles = np.percentile(y[y > 0], points)
    print(xprctiles)
    print(yprctiles)
    print(feature_intensity / xprctiles[2])
    print [feature_intensity/bg_intensity, feature_intensity, bg_intensity]

    if feature_intensity/bg_intensity < 0.75:
        return Solid
    elif feature_intensity / xprctiles[2] > 1.2:
        return Shaded
    else:
        return Empty



def id_attributes(img):
    features = get_features(img.copy())
    shape = guess_shape(features)
    color = guess_color(img.copy(),features)
    shading = guess_shading(img.copy(),features)
    return Card(color=color,count=len(features),shape=shape,shading=shading)


class CardRepo(object):
    def __init__(self,root):
        self.root = root

    def get(self, i):
        filename = os.path.join(self.root,'{:}.png'.format(i))
        if os.path.isfile(filename):
            return cv2.imread(filename)
        else:
            raise Exception('{:} not found'.format(filename))