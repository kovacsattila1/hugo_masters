import os
import torch as T
import torch.nn.functional as F
import numpy as np
from .buffer import ReplayBuffer
from .networks import ActorNetwork, CriticNetwork

class Agent():
    def __init__(self, id, alpha, beta, input_dims, tau, env,
            gamma=0.99, update_actor_interval=2, warmup=1000,
            n_actions=2, max_size=1000000, fc1_dims=400,
            fc2_dims=300, fc3_dims=200, batch_size=100, noise=0.1, chkpt_dir=''):
        self.gamma = gamma
        self.tau = tau
        self.max_action = [1]
        self.min_action = [-1]
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval
        self.id = id
        self.alpha = alpha
        self.beta = beta
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.chkpt_dir= chkpt_dir

        self.actor = ActorNetwork(id, alpha, input_dims, fc1_dims,
                                  fc2_dims, fc3_dims, n_actions=n_actions,
                                  name='actor', chkpt_dir=self.chkpt_dir)
        self.critic_1 = CriticNetwork(id, beta, input_dims, fc1_dims,
                                      fc2_dims, fc3_dims, n_actions=n_actions,
                                      name='critic_1', chkpt_dir=self.chkpt_dir)
        self.critic_2 = CriticNetwork(id, beta, input_dims, fc1_dims,
                                      fc2_dims, fc3_dims, n_actions=n_actions,
                                      name='critic_2', chkpt_dir=self.chkpt_dir)
        self.target_actor = ActorNetwork(id, alpha, input_dims, fc1_dims,
                                         fc2_dims, fc3_dims, n_actions=n_actions,
                                         name='target_actor', chkpt_dir=self.chkpt_dir)
        self.target_critic_1 = CriticNetwork(id, beta, input_dims, fc1_dims,
                                         fc2_dims, fc3_dims, n_actions=n_actions,
                                         name='target_critic_1', chkpt_dir=self.chkpt_dir)
        self.target_critic_2 = CriticNetwork(id, beta, input_dims, fc1_dims,
                                         fc2_dims, fc3_dims, n_actions=n_actions,
                                         name='target_critic_2', chkpt_dir=self.chkpt_dir)

        self.noise = noise
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        if self.time_step < self.warmup:
            mu = T.tensor(np.random.normal(scale=self.noise, size=(self.n_actions,)))
        else:
            state = T.tensor(observation, dtype=T.float).to(self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)
        something = T.tensor(np.random.normal(scale=self.noise), dtype=T.float)
        something.to(self.actor.device)
        mu_prime = mu + something
        mu_prime = T.clamp(mu_prime, self.min_action[0], self.max_action[0])
        self.time_step += 1
        return mu_prime.cpu().detach().numpy()
    

    def choose_action_eval(self, observation):
        # if self.time_step < self.warmup:
        #     mu = T.tensor(np.random.normal(scale=self.noise, size=(self.n_actions,)))
        # else:
        state = T.tensor(observation, dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(state).to(self.actor.device)
        # something = T.tensor(np.random.normal(scale=self.noise), dtype=T.float)
        # something.to(self.actor.device)
        # mu_prime = mu + something
        # mu_prime = T.clamp(mu_prime, self.min_action[0], self.max_action[0])
        # self.time_step += 1
        return mu.cpu().detach().numpy()
    

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        done = T.tensor(done).to(self.critic_1.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.critic_1.device)
        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)

        target_actions = self.target_actor.forward(state_)
        target_actions = target_actions + \
                T.clamp(T.tensor(np.random.normal(scale=0.2)), -0.5, 0.5)
        # might break if elements of min and max are not all equal
        target_actions = T.clamp(target_actions, self.min_action[0], self.max_action[0])

        q1_ = self.target_critic_1.forward(state_, target_actions)
        q2_ = self.target_critic_2.forward(state_, target_actions)

        q1 = self.critic_1.forward(state, action)
        q2 = self.critic_2.forward(state, action)

        q1_[done] = 0.0
        q2_[done] = 0.0

        q1_ = q1_.view(-1)
        q2_ = q2_.view(-1)

        critic_value_ = T.min(q1_, q2_)

        target = reward + self.gamma*critic_value_
        target = target.view(self.batch_size, 1)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()

        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.update_actor_iter != 0:
            return

        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state))
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        critic_1_state_dict = dict(critic_1_params)
        critic_2_state_dict = dict(critic_2_params)
        actor_state_dict = dict(actor_params)
        target_actor_state_dict = dict(target_actor_params)
        target_critic_1_state_dict = dict(target_critic_1_params)
        target_critic_2_state_dict = dict(target_critic_2_params)

        for name in critic_1_state_dict:
            critic_1_state_dict[name] = tau*critic_1_state_dict[name].clone() + \
                    (1-tau)*target_critic_1_state_dict[name].clone()

        for name in critic_2_state_dict:
            critic_2_state_dict[name] = tau*critic_2_state_dict[name].clone() + \
                    (1-tau)*target_critic_2_state_dict[name].clone()

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                    (1-tau)*target_actor_state_dict[name].clone()

        self.target_critic_1.load_state_dict(critic_1_state_dict)
        self.target_critic_2.load_state_dict(critic_2_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)

    def save_models(self, extension=''):
        self.actor.save_checkpoint(extension)
        self.target_actor.save_checkpoint(extension)
        self.critic_1.save_checkpoint(extension)
        self.critic_2.save_checkpoint(extension)
        self.target_critic_1.save_checkpoint(extension)
        self.target_critic_2.save_checkpoint(extension)

    def load_models(self, extension=''):
        self.actor.load_checkpoint(extension)
        self.target_actor.load_checkpoint(extension)
        self.critic_1.load_checkpoint(extension)
        self.critic_2.load_checkpoint(extension)
        self.target_critic_1.load_checkpoint(extension)
        self.target_critic_2.load_checkpoint(extension)

    
