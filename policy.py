"""
Copyright 2019-2020 @utaka233, at Sugakubunka.inc

実装時の注意
・policyは必ずclassで実装する。
・policyの行動はget_action method, 学習はfit methodで実装する。
・get_action methodはutils.plot_demo methodと揃えるため、デフォルトをgreedyにする。
"""
import numpy as np
from scipy.special import softmax
from environment import EnvSensor
from utils import binning_state, val_cumulative_reward

"""
random policy : stateに依存せず次の行動を決める。
"""
class RandomPolicy():
    def __init__(self, p = 0.5):
        self.p = p
        
    def get_action(self, state):
        action = np.random.binomial(n = 1, p = self.p, size = 1)    # stateに依存せずに行動が決まる。
        return action
    

"""
value iteration : Sarsa/Q-learningに基づく学習algorithmの実装
"""
class ValueIteration():
    def __init__(self, method):
        self.Q_table = np.random.rand(17, 2)    # stateを17 categoriesに分割（binning_state methodを参照）
        self.fit_counter = np.zeros(shape = (17, 2))
        self.state_history = []
        self.action_history = []
        self.temporal_difference_error = []
        self.mean_temporal_difference_errors = []
        self.val_cumulative_rewards = []
        self.method = method    # Sarsa or Q_learningでaction_historyのログの取り方が違う。
        assert method == "Sarsa" or method == "Q_learning", "You can set methods Sarsa or Q_learning."
        
    def reset_history(self):
        self.state_history = []
        self.action_history = []
        self.temporal_difference_error = []
        
    def get_action(self, state, epsilon = 0.0):
        # ひとまずQ-tableから次の行動を計算する。
        state_index = binning_state(state)
        self.state_history.append(state_index)    # stateのbinをログに保存する。
        action_values = self.Q_table[state_index]
        action = np.argmax(action_values)
        # Q-learningの場合、このQ-tableから計算される行動をログに保存する。
        if self.method == "Q_learning":
            self.action_history.append(action)
        # ε-greedy method : 探索する場合、actionをランダムな決定に上書きする。
        if np.random.binomial(n = 1, p = epsilon, size = 1) == 1:
            action = int(np.random.binomial(n = 1, p = 0.5, size = 1))
        # Sarsaの場合、実際に起こすactionをログに保存する。
        if self.method == "Sarsa":
            self.action_history.append(action)
        return action
    
    def fit(self, reward, gamma, learning_rate, end_of_episode, val = False):
        if end_of_episode == False:
            state_pre, state = self.state_history[-2], self.state_history[-1]
            action_pre, action = self.action_history[-2], self.action_history[-1]
            temporal_difference = reward + gamma * self.Q_table[state, action] - self.Q_table[state_pre, action_pre]
            self.Q_table[state_pre, action_pre] += learning_rate * temporal_difference
            # 学習の挙動を確認するためのログを残す。（学習したQ-tableのセルとTD誤差）
            self.fit_counter[state_pre, action_pre] += 1
            self.temporal_difference_error.append(temporal_difference)
        elif end_of_episode == True:
            state, action = self.state_history[-1], self.action_history[-1]
            self.Q_table[state, action] += learning_rate * (reward - self.Q_table[state, action])
            # 学習の挙動を確認するためのログを残す。（学習したQ-tableのセル, 1 episodeでのTD誤差の絶対値の平均値）
            self.temporal_difference_error.append(reward - self.Q_table[state, action])
            self.fit_counter[state, action] += 1
            self.mean_temporal_difference_errors.append(np.mean(self.temporal_difference_error))
            self.reset_history()    # historyを初期化
            if val == True:
                self.val_cumulative_rewards.append(
                    val_cumulative_reward(env = EnvSensor(), pol = self, 
                                          steps_per_episode = 120))
            
"""
Policy Gradient : REINFORCE algorithmを用いた方策勾配法
"""
class PolicyGradient():
    def __init__(self):
        self.parameter_table = np.zeros(shape = (17, 2))
        self.fit_counter = np.zeros(shape = (17, 2))
        self.state_history = []
        self.action_history = []
        self.reward_history = np.asarray([])
        self.gradient = np.zeros(shape = (17, 2))    # 現episodeでの勾配の絶対値を保存する。
        self.mean_absolute_gradient_history = []    # 各episodeでの勾配の絶対値の平均値を保存する。
        self.val_cumulative_rewards = []
        
    def reset_history(self):
        self.state_history = []
        self.action_history = []
        self.reward_history = np.asarray([])
        self.gradient = np.zeros(shape = (17, 2))

    def get_action(self, state):
        state_index = binning_state(state)
        self.state_history.append(state_index)    # stateのbinをログに保存する。
        policy_at_state = self.parameter_table[state_index]
        action = np.random.choice(np.arange(2), p = softmax(policy_at_state))
        self.action_history.append(action)    # actionをログに保存する。
        return action
    
    def fit(self, reward, gamma, learning_rate, end_of_episode, use_baseline = True, val = False):
        if end_of_episode == False:    # rewardのログを保存するのみ。
            self.reward_history = np.append(self.reward_history, reward)
        if end_of_episode == True:    # 学習の実行
            self.reward_history = np.append(self.reward_history, reward)    # rewardのログを保存
            # 以下、ログを読み込んで学習を始める。
            state, action, reward = self.state_history, self.action_history, self.reward_history
            for i in range(len(state)):
                policy_value = softmax(self.parameter_table[state[i], :])[action[i]]
                cumulative_reward = np.sum([r * (gamma ** t) for t, r in enumerate(reward[i:])])
                if use_baseline == True:
                    baseline = np.sum([r * (gamma ** t) for t, r in enumerate(reward[:i])])
                else:
                    baseline = 0
                self.gradient[state[i], action[i]] += (cumulative_reward - baseline) * (1-policy_value)
                self.gradient[state[i], 1-action[i]] += (cumulative_reward - baseline) * (-(1-policy_value))
                self.fit_counter[state[i], action[i]] += 1    # 各パラメータの学習回数の確認ログ
            self.parameter_table += learning_rate * self.gradient
            # 学習の挙動を確認するためのログを保存
            self.mean_absolute_gradient_history.append(np.mean(np.abs(self.gradient)))
            self.reset_history()    # historyの初期化
            if val == True:
                self.val_cumulative_rewards.append(
                    val_cumulative_reward(env = EnvSensor(), pol = self, 
                                          steps_per_episode = 120))