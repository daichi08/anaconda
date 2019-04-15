import math
import numpy as np

A = [
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
]

def calc_B_T(dt, theta):
    B_T = ([[dt*math.cos(theta), dt*math.sin(theta),  0, 1, 0],
           [0                 , 0                 , dt, 0, 1]])
    return B_T

class CartRobot():
    ### コンストラクタ
    def __init__(self, x, y, th):
        #### 状態ベクトル
        self.status = [x, y, th, 0, 0]
        #### 制御周期
        self.dt     = 0.5
        self.traj   = []

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
    
def main(init_x=0.0, init_y=0.0, init_th=math.pi/2):
    cartbot = CartRobot(init_x, init_y, init_th)
    
    return cartbot

if __name__ == "__main__":
    main()