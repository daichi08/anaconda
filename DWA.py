#!/usr/bin/env python
# coding: utf-8

import numpy as np
import itertools as it
import time
import numba
from sklearn import preprocessing
mm = preprocessing.MinMaxScaler()

def calcXiDS(xiDataSet, times, dt):
    for t in range(times):                                                                                   
        xiDataSet = np.array([A.dot(xi)+calcB(dt, xi[2]).dot(voDataSet[i]) for i,xi in enumerate(xiDataSet)])
    return xiDataSet

def calcB(dt, theta):
    B = np.array([[dt*np.cos(theta),  0],
                  [dt*np.sin(theta),  0],
                  [0               , dt],
                  [1               ,  0],
                  [0               ,  1]])
    return B

if __name__ == '__main__':
    start = time.time()
    # パラメータの宣言
    R = np.deg2rad(0.0036)
    r = 0.15/2
    d = 0.66/2
    # 初期状態の宣言
    xi = np.array([[0], [0], [np.pi/2], [0], [0]])
    u  = np.array([[0], [0]])
    # DWA用変数
    model = [1, np.deg2rad(60), 10, np.deg2rad(120), 0.1, np.deg2rad(1)]
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
    N = 1000 # 消す
    goal = np.array([10, 10]) # 消す
    obst = np.array([[3, 3],[10, 10]]) # 消す
    
    for N in range(N):
        ## データセット生成
        vData = np.arange(np.max([xi[3]-model[2]*dt, -model[0]]), np.min([xi[3]+model[2]*dt,  model[0]])+model[4], model[4])
        oData = np.arange(np.max([xi[4]-model[3]*dt, -model[1]]), np.min([xi[4]+model[3]*dt,  model[1]])+model[5], model[5])
        voDataSet = np.array(list(it.product(vData, oData)))
        ## 評価値等の準備
        dataSize  = voDataSet.shape[0]
        evalO     = 2*np.ones((dataSize, 1))
        xiDataSet = calcXiDS(np.tile(np.reshape(xi.T,(-1)),(dataSize,1)), int(sampT/calcT), calcT)
        angle = np.reshape(np.arctan2(goal[1]-xiDataSet[:,1],goal[0]-xiDataSet[:,0])-xiDataSet[:,2],(dataSize,1))
        evalV = mm.fit_transform(np.reshape(voDataSet[:,0],(dataSize,1)))
        evalA = mm.fit_transform(np.pi-abs(angle-2*np.pi*np.round(angle/(2*np.pi))))
        evalO = mm.fit_transform(evalO)
        evalSum = Vw*evalV + Aw*evalA + Ow*evalO
        u = np.reshape(voDataSet[evalSum.argmax()],(2,1))
        xi = A.dot(xi)+calcB(dt, (np.reshape(xi,(-1))[2])).dot(u)
        # print(xi.T)

    ft = time.time()-start
    print(ft)

