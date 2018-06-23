import os
import numpy as np
import random
import math
from sklearn import mixture
from sklearn.externals import joblib 

def area_ratio(xs1, ys1, xs2, ys2, xo1, yo1, xo2, yo2, ws, hs, wo, ho):
    startx = min(xs1, xo1)
    endx = max(xs2, xo2)
    width = ws+wo-(endx-startx)
    starty = min(ys1, yo1)
    endy = max(ys2, yo2)
    height = hs+ho-(endy-starty)
    areas = ws*hs
    areao = wo*ho
    if width<=0 or height<=0:
        area = 0
    else:
        area = width*height
    ratio = area/(areas+areao-area)
    return ratio
    
def getdata_gmm(pairs):
    r = np.empty(shape=[0,6])
    # prevent inf
    # pairs[:, (2, 3, 6, 7)] += 0.01
    for i in range(pairs.shape[0]):
        xs = (pairs[i,0]+pairs[i,2])/2.0
        ys = (pairs[i,1]+pairs[i,3])/2.0
        ws = pairs[i,2]-pairs[i,0]
        hs = pairs[i,3]-pairs[i,1]
        xo = (pairs[i,4]+pairs[i,6])/2.0
        yo = (pairs[i,5]+pairs[i,7])/2.0
        wo = pairs[i,6]-pairs[i,4]
        ho = pairs[i,7]-pairs[i,5]
        r1 = (xo-xs)/math.sqrt(ws*hs)
        r2 = (yo-ys)/math.sqrt(ws*hs)
        r3 = math.sqrt((wo*ho)/(ws*hs))
        r4 = area_ratio(pairs[i,0],pairs[i,1],pairs[i,2],pairs[i,3],pairs[i,4],pairs[i,5],pairs[i,6],pairs[i,7],ws, hs, wo, ho)
        r5 = ws/hs
        r6 = wo/ho
        a = np.array([[r1,r2,r3,r4,r5,r6]])
        r = np.vstack((r,a))
        if not(wo > 0 and ho > 0 and ws > 0 and hs > 0):
            print('pair: {}'.format(pairs[i]))
            # assert False
    return r
    
def apply_gmm(filtered_pairs, model_dir, model_name):
    original_path = os.getcwd()
    os.chdir(model_dir)
    gmm_model = joblib.load(model_name)
    r_filtered_pairs = getdata_gmm(filtered_pairs)
    spatial_features = gmm_model.predict_proba(r_filtered_pairs)
    os.chdir(original_path)
    return spatial_features
