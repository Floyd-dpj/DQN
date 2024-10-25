import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import random
import itertools
from agent import Agent
from tqdm import tqdm
import matplotlib.pyplot as plt
# BUFFER_SIZE = 500000
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 1000
TARGET_UPDATE_FREQUENCY = 10
epsilon =0.01
minbuffer = 200

print("Gym version:", gym.__version__)
print("PyTorch version:", torch.__version__)
env = gym.make(id="CartPole-v1", render_mode="human")
# env = gym.make(id="CartPole-v1")
s, info = env.reset()
n_state = len(s)
n_action = env.action_space.n
"""Generate agents"""
agent = Agent(idx=0,
              n_input=n_state,
              n_output=n_action,
              mode='train')

n_episode = 1000
count = 0
return_list = []
# REWARD_BUFFER = np.zeros(shape=n_episode)
for i in range(10):
        for episode_i in range(int(n_episode/10)):
            step = 0
            count1 = 0
            # for episode_i in itertools.count():
            episode_reward = 0
            done = False
            while not done and step < 600:
                random_sample = random.random()
                # print(random_sample)
                if random_sample <= epsilon:
                    a = env.action_space.sample()
                else:
                    a = agent.online_net.act(s)
                s_, r, done, timelimit, info = env.step(a)  # timelimit and info are not used in this case
                step +=1
                agent.memo.add_memo(s, a, r, done, s_)
                agent.memo.size()
                count1 +=1
                s = s_
                episode_reward += r
                if done:
                    s, info = env.reset()
                    break
                if agent.memo.size() > minbuffer:
                    # Start Gradient Step
                    batch_s, batch_a, batch_r, batch_done, batch_s_ = agent.memo.sample()  # update batch-size amounts of Q
                    # Compute Targets
                    target_q_values = agent.target_net(batch_s_)
                    max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]  # 只需要计算出目标网络中状态S‘对应的Q值即可，不需要计算其动作
                    targets = batch_r + agent.GAMMA * (1 - batch_done) * max_target_q_values
                    # Compute Q_values
                    q_values = agent.online_net(batch_s)
                    a_q_values = torch.gather(input=q_values, dim=1, index=batch_a)  # 建立最大的Q值与动作的绑定
                    # Compute Loss
                    loss = nn.functional.smooth_l1_loss(a_q_values, targets)
                    # Gradient Descent
                    agent.optimizer.zero_grad()
                    loss.backward()
                    agent.optimizer.step()

                    if count % TARGET_UPDATE_FREQUENCY == 0:
                        agent.target_net.load_state_dict(agent.online_net.state_dict())
                    count +=1
                return_list.append(episode_reward)
            print(f"Episode:{episode_i},Reward:{episode_reward}")

# episode_List = list(range(len(return_list)))
# mv_return = rl_utils.moving_average(return_list,9)
# plt.plot(episode_List,mv_return)
# plt.xlabel('Episodes')
# plt.xlabel('Returns')
# plt.title("DQN")
# plt.show()
# 实现一个简单的移动平均函数

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size
# 计算移动平均
mv_return = moving_average(return_list, 9)
# 创建 episode 列表
episode_list = list(range(len(mv_return)))

# 绘制图形
plt.plot(episode_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')  # 注意这里应该是 ylabel 而不是 xlabel
plt.title('DQN')
plt.show()

