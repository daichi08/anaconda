#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math
import itertools as it
import time
import numba
from sklearn import preprocessing
mm = preprocessing.MinMaxScaler()

@numba.jit
def calcXiDS(xiDataSet, voDataSet, times, dt):
    for i, xiData in enumerate(xiDataSet):
        for t in range(times):
            B = np.array([[dt*np.cos(xiData[2]),  0],
                          [dt*np.sin(xiData[2]),  0],
                          [0                   , dt],
                          [1                   ,  0],
                          [0                   ,  1]])
            xiData = np.reshape(A.dot(np.reshape(xiData,(-1)))+B.dot(np.reshape(voDataSet[i],(-1)))*dt,(-1))
        xiDataSet[i] = xiData
    return xiDataSet

if __name__ == '__main__':
    start = time.time()
    # パラメータの宣言
    R = np.deg2rad(0.0036)
    r = 0.15/2
    d = 0.66/2
    # 初期状態の宣言
    xi = np.reshape(np.array([0, 0, np.pi/2, 0, 0]),(5,1))
    u  = np.zeros((2,1))
    # DWA用変数
    model = np.array([1, 60, 10, 120, 0.1, 1])
    obstR = 0.8
    Vw    = 0.05
    Aw    = 1
    Ow    = 0.5
    sampT = 3
    calcT = 0.1
    # その他
    A = np.array([[1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]])
    LRFRange = 4
    dt = 0.05
    N = 1 # 消す
    goal = np.array([5, 5]) # 消す
    obst = np.array([[3, 3],[10, 10]]) # 消す
    
    for N in range(1000):
        ## 速度計算
        vMin = np.max([xi[3]-model[2]*dt,  0])
        vMax = np.min([xi[3]+model[2]*dt,  model[0]])
        oMin = np.max([xi[4]-model[3]*dt, -model[1]])
        oMax = np.min([xi[4]+model[3]*dt,  model[1]])
        ## データセット生成
        vData = np.linspace(vMin, vMax, np.int((vMax-vMin+model[4])/model[4]))
        oData = np.deg2rad(np.linspace(oMin, oMax, np.int((oMax-oMin+model[5])/model[5])))
        voDataSet = np.array(list(it.product(vData, oData)))
        ## 評価値等の準備
        dataSize  = voDataSet.shape[0]
        evalO     = 2*np.ones((dataSize, 1))
        evalA     = np.zeros((dataSize, 1))
        evalV     = np.zeros((dataSize, 1))
        xiDataSet = calcXiDS(np.tile(np.reshape(xi.T,(-1)),(dataSize,1)), voDataSet, int(sampT/calcT), calcT)
        evalV = mm.fit_transform(np.reshape(voDataSet[:,0],(dataSize,1)))
        evalA = mm.fit_transform([180-abs(a%-360) if a < -180 else
                                  180-abs(a% 360) if a >  180 else
                                  180-abs(a)
                                  for a in np.reshape(np.rad2deg(np.arctan2(goal[1]-xiDataSet[:,1],goal[0]-xiDataSet[:,0])-xiDataSet[:,2]),(dataSize,1))])
        evalO = mm.fit_transform(evalO)
        evalSum = Vw*evalV + Aw*evalA + Ow*evalO
        vod = voDataSet[np.where(np.reshape(evalSum,(-1)) == np.max(evalSum))].T
        B = np.array([[dt*np.cos(xi.T[0,2]),  0],
                      [dt*np.sin(xi.T[0,2]),  0],
                      [0                   , dt],
                      [1                   ,  0],
                      [0                   ,  1]])
        xi = A.dot(xi)+B.dot(vod)
        # print(xi.T)

    ft = time.time()-start
    print(ft)

