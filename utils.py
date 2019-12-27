"""
Copyright 2019-2020 @utaka233, at Sugakubunka.inc
"""
import numpy as np
import matplotlib.pyplot as plt
import copy
from reward import Reward


def stop_episode(time):
    if time >= 120:
        return 1
    else:
        return 0

    
"""
binning_state : stateをbinningするmethod
120℃以下 : 0
120℃-135℃ : 1から15のラベル
135℃以上 : 16
"""
def binning_state(state):
    state_index = np.floor(state - 120.0)
    if state_index <= 0:
        return 0
    elif state_index >= 16:
        return 16
    else:
        return int(state_index)
    
    
def plot_demo(env, pol, n_episodes, steps_per_episode):
    pol = copy.deepcopy(pol)
    rew = Reward()
    cumulative_rewards = []    # 累積報酬のhistory
    n_success = np.zeros(shape = (n_episodes, ))    # 125℃以上130℃以下に温度を調節出来たstep数（120steps中）
    fig, (axL, axM, axR) = plt.subplots(ncols = 3, figsize = (16, 5))
    for i in range(n_episodes):
        rewards_in_episode = []
        states_history = np.asarray([])    # 現episodeでのstateのhistory
        state = env.reset()
        states_history = np.append(states_history, state)
        for j in range(steps_per_episode):
            action = pol.get_action(state)    # ε-greedyしない。
            state = env.get_state(action)
            rewards_in_episode.append(rew.get_reward(state))
            states_history = np.append(states_history, state)
            n_success[i] = np.sum((states_history >= 125) * (states_history <= 130))   # * <=> and
        cumulative_rewards.append(np.sum(rewards_in_episode))
        step_index = [j for j in range(steps_per_episode+1)]
        axL.plot(step_index, states_history, label = str(i+1) + " episodes")
    axL.set_xlabel("step")
    axL.set_ylabel("state")
    axL.axhspan(ymin = 125.0, ymax = 130.0, color = "gray", alpha = 0.3)
    axL.legend()
    axM.boxplot(cumulative_rewards)
    axM.set_xlabel("cumulative reward")
    axR.boxplot(n_success)
    axR.set_xlabel("number of success")
    #fig.show()


def val_cumulative_reward(env, pol, steps_per_episode):
    pol = copy.deepcopy(pol)
    rew = Reward()
    rewards_in_episode = []
    state = env.reset()
    for i in range(steps_per_episode):
        action = pol.get_action(state)    # ε-greedyしない。
        state = env.get_state(action)
        rewards_in_episode.append(rew.get_reward(state))
    cumulative_reward = np.sum(rewards_in_episode)
    return cumulative_reward