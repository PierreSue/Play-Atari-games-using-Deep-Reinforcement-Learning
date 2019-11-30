# Play Atari games using Deep Reinforcement Learning

Implement an agent to play Atari games using Deep Reinforcement Learning In this project, I implemented Policy Gradient, Deep Q-Learning (DQN), Double DQN, Dueling DQN, and A2C for the atari games, such as LunarLander, Assault, and Mario.

## Atari Game
Atari Game    |    LunarLander  |     Assault    | SuperMarioBros |
--------------|:------------:|:--------------------:|:-----------:|
Demo          | <img src="https://github.com/PierreSue/Play-Atari-games-using-Deep-Reinforcement-Learning/blob/master/gifs/LunarLander.gif"> | <img src="https://github.com/PierreSue/Play-Atari-games-using-Deep-Reinforcement-Learning/blob/master/gifs/Assault.gif"> | <img src="https://github.com/PierreSue/Play-Atari-games-using-Deep-Reinforcement-Learning/blob/master/gifs/Mario.gif" width="80%" height="80%"> |

## Usage
### Training
1. training policy gradient (LunarLander):
* `$ python3 main.py --train_pg`

2. training DQN (Assault) :
* `$ python3 main.py --train_dqn`

3. training SuperMarioBros:
* `$ python3 main.py --train_mario`

### Testing
1. testing policy gradient:
* `$ python3 test.py --test_pg`

2. testing DQN:
* `$ python3 test.py --test_dqn`

3. testing SuperMarioBros:
* `$ python3 test.py --test_mario`

If "--do_render" is adopted in the command, the code can not only generate the testing score but also render the video of the corresponding atari game.

### plot figure
1. Plot the policy gradient learning curve:
* `$ cd plot`
* `$ python3 pg_plot.py`

2. Plot the DQN learning curve:
* `$ cd plot`
* `$ python3 dqn_plot.py`

3. Plot the learning curve of the comparison:
* `$ cd plot`
* `$ python3 dqn_4plot.py`

4. Plot the learning curve of the improvement model:
* `$ cd plot`
* `$ python3 improvement_plot.py`

5. Plot the learning curve of leaning mario:
* `$ cd plot`
* `$ python3 mario_plot.py`

## Results
### 1. Policy Gradient

* Algorithm

    <img src="https://github.com/PierreSue/Play-Atari-games-using-Deep-Reinforcement-Learning/blob/master/plot/PG-Alg.png"> 

    I first used PolicyNet to get the probability from the current state, and used torch.distributions.Categorical to sample an action from the probability. Then, by discounting the saved loss and using the loss function shown above, the model can update the parameters of the PolicyNet.

* Learning Curve

    <img src="https://github.com/PierreSue/Play-Atari-games-using-Deep-Reinforcement-Learning/blob/master/plot/pg.png" width="50%" height="50%"> 

### 2. DQN

* Algorithm

    <img src="https://github.com/PierreSue/Play-Atari-games-using-Deep-Reinforcement-Learning/blob/master/plot/DQN-Alg.png" width="50%" height="50%"> 

    There are two PolicyNet used here. I first used the online_net to get the probability from the responding action. If exploration occurred, I randomly sampled an action. Otherwise, I sampled an action like what PG model had done. The exploration schedule here is 1.0 + (1.0 – 0.01) * math.exp(-1. * current_step / 3000000). So the model would tend to explore at first and gradually used the model more to predict. Then, by sampling the buffer, discounting the saved loss and using the loss function shown above, the model can update the parameters of the PolicyNet.

* Learning Curve

    <img src="https://github.com/PierreSue/Play-Atari-games-using-Deep-Reinforcement-Learning/blob/master/plot/dqn.png" width="50%" height="50%">

* Hyper-parameter Tuning

     Original:Exploration epsilon = 1.0 – 0.01 ; gamma = 0.99

     Exploration_H:Exploration epsilon = 1.0 – 0.7 ; gamma = 0.99 

     Exploration_L:Exploration epsilon = 0.3 – 0.01 ; gamma = 0.99

     Exploration_M:Exploration epsilon = 0.3 – 0.7 ; gamma = 0.99

    <img src="https://github.com/PierreSue/Play-Atari-games-using-Deep-Reinforcement-Learning/blob/master/plot/dqn4.png" width="50%" height="50%">

    As for the original model, the result is not bad by the way, but there are still better models for this task. As for the Exploration_H model, it doesn’t work well because it doesn’t take too much the PolicyNet model prediction into account. As for the Exploration_L model, it works surprisingly well here, and I think the reason here is that exploration is rather less necessary for this task because Assault is quite a straight- forward task. And as for the Exploration_M model, little change in epsilon (nearly 0.5) makes fluctuations at the training step, which directly influences the average reward at testing step.

* Other Q-learning Algorithm

     Double DQN

    <img src="https://github.com/PierreSue/Play-Atari-games-using-Deep-Reinforcement-Learning/blob/master/plot/Double-DQN-Alg.png" width="50%" height="50%">

    The advantage of double-dqn is that it uses two networks to decouple the action selection from the target Q value generation when we compute the Q target, which helps us reduce the overestimation of q values and, as a consequence, helps us train faster and have more stable learning to get better results.

     Dueling DQN

    <img src="https://github.com/PierreSue/Play-Atari-games-using-Deep-Reinforcement-Learning/blob/master/plot/Dueling-DQN-Alg.png" width="30%" height="30%">

    The advantage of dueling-dqn is that it decomposes Q-function into advantage and value function, which avoids the network being overoptimistic. Otherwise it will cause actual DQN and to get as optimistic as it explodes.

* Comparison

    <img src="https://github.com/PierreSue/Play-Atari-games-using-Deep-Reinforcement-Learning/blob/master/plot/improvement.png" width="50%" height="50%">

3. SuperMario

* Algorithm

    <img src="https://github.com/PierreSue/Play-Atari-games-using-Deep-Reinforcement-Learning/blob/master/plot/A2C-Alg.png" width="80%" height="80%"> 

    I use A2C(policy gradient along with dqn) in my implementation. TD and entropy regularization are used here, but I don’t use RNN in this A2C model.

* Learning Curve

    <img src="https://github.com/PierreSue/Play-Atari-games-using-Deep-Reinforcement-Learning/blob/master/plot/mario.png" width="50%" height="50%">
