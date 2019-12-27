"""
Copyright 2019-2020 @utaka233, at Sugakubunka.inc
"""
import numpy as np

class EnvSensor():
    def __init__(self):
        self.state = None    # 状態
    
    def reset(self):
        self.state = np.random.normal(loc = 120.0, scale = 10.0, size = 1)    # 初期温度 : 120±10℃
        return self.state
        
    def get_state(self, action):
        random_effect = np.random.normal(loc = 2.0, scale = 1.0, size = 1)    # 自然な温度上昇
        action_effect = np.random.normal(loc = -3.0, scale = 0.5, size = 1) * action    # 行動による冷却
        self.state = self.state + random_effect + action_effect    # 行動後の状態
        return self.state