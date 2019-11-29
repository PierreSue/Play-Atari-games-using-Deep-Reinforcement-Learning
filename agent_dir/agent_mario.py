import torch
from torch.distributions import Categorical
from torch.optim import RMSprop
from torch.nn.utils import clip_grad_norm_

from a2c.environment_a2c import make_vec_envs
from a2c.storage import RolloutStorage
from a2c.actor_critic import ActorCritic

from collections import deque
import os

use_cuda = torch.cuda.is_available()

class AgentMario:
    def __init__(self, env, args):

        # Hyperparameters
        self.lr = 7e-4
        self.gamma = 0.9
        self.hidden_size = 512
        self.update_freq = 5
        self.n_processes = 64
        self.seed = 7122
        self.max_steps = 1e7
        self.grad_norm = 0.5
        self.entropy_weight = 0.05

        #######################    NOTE: You need to implement
        self.recurrent = False # <- ActorCritic._forward_rnn()
        #######################    Please check a2c/actor_critic.py
        
        self.display_freq = 4000
        self.save_freq = 100000
        self.save_dir = './checkpoints/'

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
         
        self.envs = env
        if self.envs == None:
            self.envs = make_vec_envs('SuperMarioBros-v0', self.seed,
                    self.n_processes)
        self.device = torch.device("cuda:1" if use_cuda else "cpu")

        self.obs_shape = self.envs.observation_space.shape
        self.act_shape = self.envs.action_space.n

        self.rollouts = RolloutStorage(self.update_freq, self.n_processes,
                self.obs_shape, self.act_shape, self.hidden_size) 
        self.model = ActorCritic(self.obs_shape, self.act_shape,
                self.hidden_size, self.recurrent).to(self.device)
        self.optimizer = RMSprop(self.model.parameters(), lr=self.lr, 
                eps=1e-5)

        if args.test_mario:
            self.load_model(os.path.join('mario.pt'))
            print('finish model loading ...')

        self.hidden = None
        self.init_game_setting()
   
    def _update(self):
        # TODO: Compute returns
        # R_t = reward_t + gamma * R_{t+1}
        rewards = self.rollouts.rewards
        obs = self.rollouts.obs
        hiddens = self.rollouts.hiddens
        masks = self.rollouts.masks
        actions = self.rollouts.actions
        preds = self.rollouts.value_preds

        # 5 x 16 x 1
        Vt = preds[:-1]
        Vt_1 = self.gamma * preds[1:] * masks[:-1]
        
        # 5 x 16
        from torch.autograd import Variable
        Advantage = Variable((rewards - (Vt-Vt_1)), requires_grad=False)
        R = Advantage.squeeze(-1)

        # TODO:
        # Compute actor critic loss (value_loss, action_loss)
        # OPTIONAL: You can also maxmize entropy to encourage exploration
        # loss = value_loss + action_loss (- entropy_weight * entropy)
        entropys = []
        logP = []
        Q_values = []

        for idx, (ob, hidden, mask) in enumerate(zip(obs, hiddens, masks)):
            value, action_prob, _ = self.model(ob, hidden, mask)
            Q_values.append(value)
            if idx != obs.size(0)-1:
                m = Categorical(action_prob)
                logP.append(m.log_prob(actions[idx].squeeze(-1)))
                entropys.append(torch.mean(m.entropy()))

        logP = torch.stack(logP,0)
        action_loss = torch.mean(-R * logP)
        print(action_loss)

        Q_values = torch.stack(Q_values, 0)
        Qt = Q_values[:-1]
        Qt_1 = rewards + self.gamma * preds[1:] * masks[:-1]

        mse = torch.nn.MSELoss()
        value_loss = mse(Qt, Qt_1)
        print(value_loss)

        entropys = sum(entropys)/len(entropys)
        print(entropys)
        loss = value_loss + action_loss - entropys
        print(loss)

        # Update
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.grad_norm)
        self.optimizer.step()
        
        # TODO:
        # Clear rollouts after update (RolloutStorage.reset())
        self.rollouts.reset()

        return loss.item()

    def _step(self, obs, hiddens, masks):
        from torch.autograd import Variable
        from torch.distributions import Categorical
        import numpy as np
        
        with torch.no_grad():
            # TODO:
            # Sample actions from the output distributions
            # HINT: you can use torch.distributions.Categorical
            values, action_probs, hiddens = self.model(obs, hiddens, masks)
            m = Categorical(action_probs)
            actions = m.sample()

        obs, rewards, dones, infos = self.envs.step(actions.cpu().numpy())
        
        # TODO:
        # Store transitions (obs, hiddens, actions, values, rewards, masks)
        # You need to convert arrays to tensors first
        # HINT: masks = (1 - dones)
        obs      = Variable(torch.FloatTensor(np.float32(obs)))
        rewards  = Variable(torch.FloatTensor(np.float32(rewards)))
        dones    = Variable(torch.FloatTensor(np.float32(dones))).unsqueeze(1)
        masks    = torch.ones(masks.shape) - dones
        self.rollouts.insert(obs, hiddens, actions.unsqueeze(-1), values, rewards.unsqueeze(-1), masks)

        
    def train(self):
        # logging
        import logging
        logging.basicConfig(filename="mario_reward.log", level=logging.INFO)

        print('Start training')
        running_reward = deque(maxlen=10)
        episode_rewards = torch.zeros(self.n_processes, 1).to(self.device)
        total_steps = 0
        
        # Store first observation
        obs = torch.from_numpy(self.envs.reset()).to(self.device)
        self.rollouts.obs[0].copy_(obs)
        self.rollouts.to(self.device)
        
        while True:
            # Update once every n-steps
            for step in range(self.update_freq):
                self._step(
                    self.rollouts.obs[step],
                    self.rollouts.hiddens[step],
                    self.rollouts.masks[step])

                # Calculate episode rewards
                episode_rewards += self.rollouts.rewards[step]
                for r, m in zip(episode_rewards, self.rollouts.masks[step + 1]):
                    if m == 0:
                        running_reward.append(r.item())
                episode_rewards *= self.rollouts.masks[step + 1]

            loss = self._update()
            total_steps += self.update_freq * self.n_processes

            # Log & save model
            if len(running_reward) == 0:
                avg_reward = 0
            else:
                avg_reward = sum(running_reward) / len(running_reward)

            if total_steps % self.display_freq == 0:
                logging.info("{},{}".format(total_steps, avg_reward))
                print('Steps: %d/%d | Avg reward: %f'%
                        (total_steps, self.max_steps, avg_reward))
            
            if total_steps % self.save_freq == 0:
                self.save_model('model.pt')
            
            if total_steps >= self.max_steps:
                break

    def save_model(self, filename):
        torch.save(self.model, os.path.join(self.save_dir, filename))

    def load_model(self, path):
        self.model = torch.load(path, map_location=torch.device('cpu'))

    def init_game_setting(self):
        if self.recurrent:
            self.hidden = torch.zeros(1, self.hidden_size).to(self.device)

    def make_action(self, observation, test=False):
        # TODO: Use you model to choose an action
        from torch.autograd import Variable
        
        observation = Variable(torch.from_numpy(observation).float().unsqueeze(0)).to(self.device)
        value, action_prob, hidden = self.model(observation, observation, observation)
        m = Categorical(action_prob)
        action = torch.argmax(m.probs).data.tolist()
        return action
