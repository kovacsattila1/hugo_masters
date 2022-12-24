import gym
import numpy as np
from dddqn_agent import Agent
from utils import plot_learning_curve
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if __name__ == '__main__':
  env = gym.make('LunarLander-v2')
  agent = Agent(lr=0.0005, gamma=0.99, n_actions=4, epsilon=1.0, batch_size=64, input_dims=[8])
  # agent.load_networks()
  n_games = 500
  scores = []
  eps_history = []

  for i in range(n_games):
    done = False
    score = 0
    observation = env.reset()
    while not done:
      action = agent.choose_action(observation)
      observation_, reward, done, info = env.step(action)
      score += reward
      agent.store_transition(observation, action, reward, observation_, done)
      observation = observation_
      agent.learn()
      # env.render()
    eps_history.append(agent.epsilon)
    scores.append(score)
    avg_score = np.mean(scores[-30:])
    print('episode ', i, 'score %.1f' % score, 'avg score %.1f' % avg_score)

  #saving models weights
  agent.q_eval.save_weights('./q_eval_weights/')
  agent.q_next.save_weights('./q_next_weights/')

  filename = 'lunar_lander_dueling_ddqn.png'
  x = [i + 1 for i in range(n_games)]
  plot_learning_curve(x, scores, eps_history, filename)


