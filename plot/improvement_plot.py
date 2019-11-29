import matplotlib.pyplot as plt
from pylab import *

def plot_reward(file_name):
    with open(file_name, 'r') as ip:
        lines = ip.readlines()
        step = []
        reward = []
        for idx, line in enumerate(lines):
            line = line.split(':')[-1]
            line = line.split(',')
            step.append(int(line[0]))
            reward.append(float(line[1]))
    plot(step, reward, label=file_name.split('.')[0])


plot_reward('dqn_reward.log')
plot_reward('double_reward.log')
plot_reward('dueling_reward.log')
plt.legend(loc='upper right')
plt.savefig('improvement.png')
