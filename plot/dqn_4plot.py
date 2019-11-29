import matplotlib.pyplot as plt
from pylab import *

def plot_reward(file_name):
    with open(file_name, 'r') as ip:
        lines = ip.readlines()
        step = []
        reward = []
        last = len(lines)*0.8
        for idx, line in enumerate(lines):
            if idx > last:
                line = line.split(':')[-1]
                line = line.split(',')
                step.append(int(line[0]))
                reward.append(float(line[1]))
    plot(step, reward, label=file_name.split('.')[0])


plot_reward('original.log')
plot_reward('exploration_H.log')
plot_reward('exploration_L.log')
plot_reward('exploration_M.log')
plt.legend(loc='upper right')
plt.savefig('dqn4.png')
