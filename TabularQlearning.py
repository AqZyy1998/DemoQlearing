import logging

import numpy as np
import pandas as pd
import time

N_STATE = 6  # 宽度
ACTIONS = ['left', 'right']
EPSILON = 0.9  # greedy
ALPHA = 0.1  # 学习率
GAMMA = 0.9  # γ衰减值
MAX_EPISODES = 13  # 最大回合数
FRESH_TIME = 0.3  # 移动时间间隔


# Q表 n x 2(left, right)
def build_q_table(n_states, actions):
    table = pd.DataFrame(np.zeros((n_states, len(actions))), columns=actions)
    return table


# 选择action的policy
def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]  # 选出state所有action值
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):  # 非贪婪 or 或者这个 state 还没有探索过
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.idxmax()  # 贪婪模式 argmax有点问题 新numpy中用idxmax
    return action_name


# 反馈S_ R
def get_env_feedback(state, action):
    if action == 'right':
        if state == N_STATE - 2:  # N_STATE - 2即T的位置
            state_ = 'terminal'
            reward = 1
        else:
            state_ = state + 1
            reward = 0
    else:
        reward = 0
        if state == 0:
            state_ = state
        else:
            state_ = state - 1
    return state_, reward


# 环境更新
def update_env(state, episode, step_counter):
    env_list = ['-'] * (N_STATE - 1) + ['T']  # '-----T'
    if state == 'terminal':
        interaction = 'episode %s: total_steps = %s' % (episode + 1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(1)
        print('\r      ', end='')
    else:
        env_list[state] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    q_table = build_q_table(N_STATE, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        state = 0
        is_terminated = False
        update_env(state, episode, step_counter)  # 初始化环境
        while not is_terminated:
            action = choose_action(state, q_table)
            state_, reward = get_env_feedback(state, action)
            q_predict = q_table.loc[state, action]  # 估计值
            if state_ != 'terminal':
                q_real = reward + GAMMA * q_table.iloc[state_, :].max()  # 真实值
            else:  # 触发终止条件
                q_real = reward
                is_terminated = True

            q_table.loc[state, action] += ALPHA * (q_real - q_predict)  # 更新q表
            state = state_  # 更新位置
            update_env(state, episode, step_counter + 1)
            step_counter += 1
    return q_table


if __name__ == '__main__':
    q_table = rl()
    print(q_table)
