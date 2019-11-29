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
    plot(step, reward)


plot_reward('mario_reward.log')
plt.savefig('mario.png')
