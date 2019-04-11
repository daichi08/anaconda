#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math
import time

# 関数群
## 入力にかかる行列計算用
def calc_B_T(dt, theta):
    B_T = ([[dt*math.cos(theta), dt*math.sin(theta),  0, 1, 0],
           [0                 , 0                 , dt, 0, 1]])
    return B_T
## 角度補正用
def correction_ang(angle):
    if angle > math.pi:
        while angle > math.pi:
            angle -= 2*math.pi
    elif angle < -math.pi:
        while angle < -math.pi:
            angle += 2*math.pi

    return angle

# 変数群
## 状態ベクトルにかかる行列
A = [
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
]

# クラス群

## シミュレーション用ロボットモデル
class SimRobotModel():
    ### コンストラクタ
    def __init__(self):
        #### 速度に関する制限
        self.max_vel = 1.4
        self.min_vel = 0.0
        self.max_acc = 9.0

        #### 回転速度に関する制限
        self.max_ang_vel =  60 * math.pi/180
        self.min_ang_vel = -60 * math.pi/180
        self.max_ang_acc = 120 * math.pi/180

    ### 取りうる状態ベクトルの計算
    def predict_status(self, u, status, dt, pre_step):
        next_status = []

        for i in range(pre_step):
            B_T = calc_B_T(dt, status[2])
            status = list(
                np.dot(status, A) + np.dot(u, B_T)
            )
            next_status.append(status)
        return next_status

## Dynamic Window Approachのコントローラ
class DWA():
    ### コンストラクタ
    def __init__(self):
        #### モデル呼び出し
        self.simbot = SimRobotModel()

        #### 分解能
        self.delta_vel     = 0.1
        self.delta_ang_vel = math.pi/180

        #### シミュレーションの時間、間隔およびステップの宣言
        self.pre_time  = 2
        self.samp_time = 0.1
        self.pre_step  = int(self.pre_time/self.samp_time)

    ### 取りうる入力の組み合わせ全探索
    def _calc_u_set(self, status):
        calc_max_vel     = self.simbot.max_acc*self.samp_time
        calc_max_ang_vel = self.simbot.max_ang_acc*self.samp_time
        v_set = np.arange(
            max(
                status[3]-calc_max_vel,
                self.simbot.min_vel
            ),
            min(
                status[3]+calc_max_vel,
                self.simbot.max_vel
            ),
            self.delta_vel
        )
        o_set = np.arange(
            max(
                status[4]-calc_max_ang_vel,
                self.simbot.min_ang_vel
            ),
            min(
                status[4]+calc_max_ang_vel,
                self.simbot.max_ang_vel
            ),
            self.delta_ang_vel
        )
        return v_set, o_set

    ### 取りうる経路の計算
    def make_path(self, status):
        v_set, o_set = self._calc_u_set(status)
        paths = []

        for vel in v_set:
            for ang_vel in o_set:
                path = self.simbot.predict_status([vel, ang_vel], status, self.samp_time, self.pre_step)
                paths.append(path)

        return paths

## メイン制御
class MainController():
    ### コンストラクタ
    def __init__(self):
        self.controller = DWA()

    ### 走行中の処理
    def runnning(self, status):
        paths = self.controller.make_path(status)

        return paths

def main(status):
    controller = MainController()
    paths = controller.runnning(status)
    return paths
    
if __name__ == '__main__':
    start = time.time()
    status = main()
    finished = time.time() -start
    print("{0}".format(finished) + "[sec]")
    print(status)