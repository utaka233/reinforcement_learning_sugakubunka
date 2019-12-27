"""
Copyright 2019-2020 @utaka233, at Sugakubunka.inc
"""
import numpy as np

class Reward():
    def __init__(self):
        self.reward_ = 0
    
    def get_reward(self, state):
        if state >= 125 and state <= 130:
            self.reward_ = 1
        else:
            self.reward_ = -1
        return self.reward_