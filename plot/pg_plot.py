import matplotlib.pyplot as plt
from pylab import *

with open('./pg_reward.log', 'r') as ip:
    lines = ip.readlines()
    step = []
    reward = []
    for idx, line in enumerate(lines):
        line = line.split(':')[-1]
        line = line.split(',')
        step.append(int(line[0]))
        reward.append(float(line[1]))

plot(step, reward)
plt.savefig('pg.png')
