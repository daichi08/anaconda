
# coding: utf-8

# In[23]:


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

def normalize(data):
    data = np.array(data)
    max_data = max(data)
    min_data = min(data)

    if max_data == min_data:
        data = [0.0 for i in range(len(data))]
    else:
        data = (data - min_data) / (max_data - min_data)

    return data

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
## 実際のロボット用
class CartRobot():
    ### コンストラクタ
    def __init__(self, init_x, init_y, init_th):
        #### 状態ベクトル
        self.status = [init_x, init_y, init_th, 0.0, 0.0]
        #### 軌跡記録用
        self.traj   = []
        #### 制御周期
        self.dt     = 0.5
        #### 機体パラメータ(ROSに移行する場合は消す)
        self.u_r = 0
        self.u_l = 0
        self.u_r_hist = [self.u_r]
        self.u_l_hist = [self.u_l]

    ### 状態ベクトルの更新用
    def update_status(self, u):
        B_T = calc_B_T(self.dt, self.status[2])
        next_status = list(
            np.dot(self.status, A) +
            np.dot(u, B_T)
        )
        self.traj.append(self.status)
        self.status = next_status

        return next_status

    ### 周波数への変換(ROSに移行する場合は切り分ける)
    def calc_freq(self, v, omega):
        R = 0.0036 * math.pi/180
        r = 0.15/2
        d = 0.66/2

        ### v = R*r/2 * (u_r + u_l), omega = R*r/(2*d) * (u_r - u_l)で計算
        u_r = int((v+d*omega)/(R*r))
        u_l = int((v-d*omega)/(R*r))

        self.u_r = u_r
        self.u_l = u_l
        self.u_r_hist.append(u_r)
        self.u_l_hist.append(u_l)

        return u_r, u_l

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

        #### 評価用重み
        self.weight_ang = 0.8
        self.weight_vel = 0.4
        self.weight_obs = 0.5

        #### 経路記録用
        self.traj_opt   = []

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
    def _make_path(self, status):
        v_set, o_set = self._calc_u_set(status)
        paths = []

        for vel in v_set:
            for ang_vel in o_set:
                path = self.simbot.predict_status([vel, ang_vel], status, self.samp_time, self.pre_step)
                paths.append(path)

        return paths

    ### 入力の決定
    def calc_input(self, goal, status, obstacles):
        paths = self._make_path(status)
        opt_path = self._eval_path(paths, goal, status, obstacles)
        self.traj_opt.append(opt_path)

        return paths, opt_path

    ### 評価値の生成
    def _eval_path(self, paths, goal, status, obstacles):
        nearest_obs = self._calc_obs_dist(status, obstacles)

        score_angs = []
        score_vels = []
        score_obs  = []

        for path in paths:
            score_angs.append(self._angs(path, goal[0], goal[1]))
            score_vels.append(self._vels(path))
            score_obs.append(self._obs(path, nearest_obs))

        normalize_angs = np.array(normalize(score_angs))
        normalize_vels = np.array(normalize(score_vels))
        normalize_obs  = np.array(normalize(score_obs))

        total_score = self.weight_ang * normalize_angs +                      self.weight_vel * normalize_vels +                      self.weight_obs * normalize_obs
        max_score_index = list(total_score).index(max(total_score))
        opt_path = paths[max_score_index][-1]

        return opt_path

    ### 角度評価用
    def _angs(self, path, g_x, g_y):
        last_x  = path[-1][0]
        last_y  = path[-1][1]
        last_th = path[-1][2]

        score_ang = math.pi-abs(
            correction_ang(
                math.atan2(g_y-last_y, g_x-last_x) - last_th
            )
        )

        return score_ang

    ### 速度評価用
    def _vels(self, path):
        score_vel = path[-1][3]
        return score_vel

    ### 障害物評価用
    def _obs(self, path, nearest_obs):
        score_obs = 2
        tmp_dist  = 0.0
        datasize  = len(path)

        for i in range(datasize):
            for obs in nearest_obs:
                tmp_dist = math.sqrt((path[i][0]-obs.position[0])**2 + (path[i][1]-obs.position[1])**2)

                #if tmp_dist < obs.size:
                #    score_obs = -float('inf')
                #elif tmp_dist < score_obs:
                #    score_obs = tmp_dist

        return score_obs

    ### 障害物の絞り込み
    def _calc_obs_dist(self, status, obstacles):
        dist_to_obs = 5
        nearest_obs = []

        for obs in obstacles:
            dist_obs = math.sqrt((status[0] - obs.position[0]) ** 2 + (status[1] - obs.position[1]) ** 2)
            if dist_obs < dist_to_obs:
                nearest_obs.append(obs)

        return nearest_obs

## ゴール生成用(人物の位置に置き換える)
class Goal():
    ### コンストラクタ
    def __init__(self):
        #### ゴール位置
        self.position = [10, 10]
        #### ゴール位置の記録用
        self.traj_position = []

    def update_position(self):
        self.traj_position.append(self.position)
        return self.position

## 障害物生成用(LRFデータに置き換える)
class Obstacle():
    ### コンストラクタ
    def __init__(self, x, y, size = 0.8):
        self.position = [x, y]
        self.size = size

## メイン制御
class MainController():
    ### コンストラクタ
    def __init__(self):
        self.cartbot    = CartRobot(0, 0, math.pi/2)
        self.const_goal = Goal()
        self.controller = DWA()
        self.obstacles  = [Obstacle(3, 3), Obstacle(5,5), Obstacle(8,4)]

    ### 走行中の処理
    def runnning(self):
        goal_flg = False
        max_step  = 100

        # while not goal_flg:
        for n in range(max_step):
            goal  = self.const_goal.update_position()
            paths, opt_path = self.controller.calc_input(goal, self.cartbot.status, self.obstacles)
            u = opt_path[3:5]
            self.cartbot.update_status(u)
            #### ROSの場合は消す
            u_r, u_l = self.cartbot.calc_freq(u[0], u[1])

        return self.cartbot.status

def main():
    controller = MainController()
    status = controller.runnning()

    return status

if __name__ == '__main__':
    start = time.time()
    status = main()
    finished = time.time() -start
    print("{0}".format(finished) + "[sec]")
    print(status)


# In[21]:


cartbot.u_l_hist

