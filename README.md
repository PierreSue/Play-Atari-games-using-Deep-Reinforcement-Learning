# Play Atari games using Deep Reinforcement Learning

## Atari Game
Atari Game    |    LunarLander  |     Assault    | SuperMarioBros |
--------------|:------------:|:--------------------:|:-----------:|
Demo          | <img src="https://github.com/PierreSue/Play-Atari-games-using-Deep-Reinforcement-Learning/blob/master/gifs/LunarLander.gif"> | <img src="https://github.com/PierreSue/Play-Atari-games-using-Deep-Reinforcement-Learning/blob/master/gifs/Assault.gif"> | <img src="https://github.com/PierreSue/Play-Atari-games-using-Deep-Reinforcement-Learning/blob/master/gifs/Mario.gif"> |

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

