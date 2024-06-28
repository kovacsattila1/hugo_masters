#!/usr/bin/env python3

import torch.multiprocessing.queue
import rospy
from sensor_msgs.msg import Image # ROS Image message -> OpenCV2 image converter
# from cv_bridge import CvBridge, CvBridgeError # OpenCV2 for saving an image
# import cv2
import math
from std_msgs.msg import Float32
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Bool
from std_msgs.msg import Int32
from geometry_msgs.msg import Point32
import time
from multiprocessing import Process, Queue
import signal
import _thread
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import random
import matplotlib.pyplot as plt
import numpy as np
from ddpg import DDPG
import logging
import torch
from utils.replay_memory import ReplayMemory, Transition
from utils.noise import OrnsteinUhlenbeckActionNoise
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import os
from functools import partial
from DDPG.ddpg_torch import Agent as Agent_ddpg
from DDPG.utils import plot_learning_curve as plc_ddpg

from TD3.td3_torch import Agent as Agent_td3
from TD3.utils import plot_learning_curve as plc_td3

import subprocess
# import multiprocessing
import sys

import re

import csv


cold_start = False

got_state = False


counter = 0
prev_state = 0

original_pos = []
original_ori = []


logger = logging.getLogger('train')
logger.setLevel(logging.DEBUG)

# Instantiate CvBridge
# bridge = CvBridge()


got_position = False
got_simstate = False



time_list = [0]
pos_x_list = [0]
pos_y_list = [0]
pos_z_list = [0]
ori_x_list = [0]
ori_y_list = [0]
ori_z_list = [0]

actual_joint_positions = []
actual_pos = []
actual_ori = []
actual_time = 0
step_cb_enable = False

prev_pos = []
prev_ori = []
prev_joint_positions = []

dt = 0


#------------------------------------------------------------------------------------------------------------

# def process_image(q):

#     def image_cb(msg):
#         #print("Received an image!")
#         try:
#             # Convert your ROS Image message to OpenCV2
#             cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
#             cv2_img = cv2.flip(cv2_img, 0)
#         except CvBridgeError as e:
#             print(e)
#         else:
#             # Save your OpenCV2 image as a jpeg 
#             #cv2.imwrite('camera_image.jpeg', cv2_img)
#             cv2.namedWindow("Input")
#             cv2.imshow("Input", cv2_img)
#             cv2.waitKey(1)


#     rospy.init_node('hugo_vision')
#     q_size = 10
#     rospy.Subscriber("/image", Image, image_cb, queue_size=q_size, buff_size=2**10)

#     rate = rospy.Rate(10)
#     while not rospy.is_shutdown():
#         rate.sleep()



#------------------------------------------------------------------------------------------------------------



def joint_control_td3(m2s, s2m, pid, file_path, id):

    with open(file_path, 'w') as f:
        sys.stdout = f

        class Coppelia:
            def __init__(self):
                # rospy.init_node('hugo_vision' + str(id))
                self.observation_space  = torch.tensor(156*[0])
                self.action_space = torch.tensor(20 * [0])

            def reset(self):
                # print("reset called")
                os.kill(pid, signal.SIGALRM)
                state = m2s.get()
                return state
            

        env = Coppelia()

        agent = Agent_td3(id, alpha=0.001, beta=0.001, 
                        input_dims=env.observation_space.shape, tau=0.005, env=env,
                        batch_size=100, fc1_dims=800, fc2_dims=600, fc3_dims=400,
                        n_actions=env.action_space.shape[0])
        n_games = 15000
        filename = "" \
            + 'Coppelia' + '_' \
            + 'td3' + '_' \
            + 'alpha' + str(agent.alpha) + '_' \
            + 'beta' +  str(agent.beta) + '_' \
            + 'games' + str(n_games) + '_' \
            + 'fc1_' + str(agent.fc1_dims) + '_' \
            + 'fc2_' + str(agent.fc2_dims) + '_' \
            + 'fc3_' + str(agent.fc3_dims)
        
        figure_file = 'plots/' + filename + '_' + str(id) + '.png'

        best_score = -500
        score_history = []




        for i in range(n_games):
                observation = env.reset()
                # print("first observation", observation, flush=True)
                done = False
                score = 0

                while not done:
                    action = agent.choose_action(observation)
                    s2m.put([torch.tensor(action)])
                    # print(env.step(action))
                    # observation_, reward, done, _, _ = env.step(action)

                    observation_ = m2s.get()
                    # print("other observation", observation, flush=True)
                    reward = m2s.get()
                    done = m2s.get()


                    agent.remember(observation, action, reward, observation_, done)
                    agent.learn()
                    # print("reward", reward, flush=True)
                    score += reward
                    observation = observation_

                score_history.append(score)
                avg_score = np.mean(score_history[-100:])

                if avg_score > best_score:
                    best_score = avg_score
                    agent.save_models()

                print('episode ', i, 'score %.1f' % score,
                        'trailing 100 games avg %.1f' % avg_score, flush=True)

        x = [i+1 for i in range(n_games)]
        plc_td3(x, score_history, figure_file)



#------------------------------------------------------------------------------------------------------------



def joint_control_ddpg(m2s, s2m, pid, controller_output_file_path, controller_output_folder_path,  id):
    print("Controller process with id = ", id, "started", flush=True)
    # print("debug - controller output file path = ", controller_output_file_path)

    with open(controller_output_file_path, 'w') as f:
        sys.stdout = f
        print("File output testing", flush=True)
        class Coppelia:
            def __init__(self):
                # rospy.init_node('hugo_vision' + str(id))
                self.observation_space  = torch.tensor(156*[0])
                self.action_space = torch.tensor(20 * [0])

            def reset(self):
                # print("reset called")
                os.kill(pid, signal.SIGALRM)
                state = m2s.get()
                return state
            

        env = Coppelia()

        agent = Agent_ddpg(id, alpha=0.0001, beta=0.001, 
                        input_dims=env.observation_space.shape, tau=0.001,
                        batch_size=64, fc1_dims=800, fc2_dims=600, fc3_dims=400,
                        n_actions=env.action_space.shape[0], chkpt_dir=controller_output_folder_path)
        n_games = 12000
        filename = "" \
            + 'Coppelia' + '_' \
            + 'ddpg' + '_' \
            + 'alpha' + str(agent.alpha) + '_' \
            + 'beta' +  str(agent.beta) + '_' \
            + 'games' + str(n_games) + '_' \
            + 'fc1_' + str(agent.fc1_dims) + '_' \
            + 'fc2_' + str(agent.fc2_dims) + '_' \
            + 'fc3_' + str(agent.fc3_dims)
        
        figure_file = controller_output_folder_path + "/plots/" + filename + '_' + str(id) + '.png'
        step_rewards_file = controller_output_folder_path + "/rewards/step_rewards/" + filename + '_' + str(id) + '.csv'
        episode_rewards_file = controller_output_folder_path + "/rewards/episode_rewards/" + filename + '_' + str(id) + '.csv'
        average_rewards_file = controller_output_folder_path + "/rewards/average_rewards/" + filename + '_' + str(id) + '.csv'

        #step_rewards_figure = controller_output_folder_path + "/rewards/step_rewards/" + filename + '_' + str(id) + '.csv'
        #episode_rewards_figure = controller_output_folder_path + "/rewards/episode_rewards/" + filename + '_' + str(id) + '.csv'
        #average_rewards_figure = controller_output_folder_path + "/rewards/average_rewards/" + filename + '_' + str(id) + '.csv'

        best_score = -1000
        score_history = []
        step_reward_history = []
        episode_reward_history = []
        average_reward_history = [] 


        for i in range(n_games):
            observation = env.reset()

            done = False
            score = 0
            agent.noise.reset()
            while not done:
                action = agent.choose_action(observation)
                s2m.put([torch.tensor(action)])

                observation_ = m2s.get()
                reward = m2s.get()
                step_reward_history.append(reward)
                done = m2s.get()

                agent.remember(observation, action, reward, observation_, done)
                agent.learn()
                score += reward
                observation = observation_

            episode_reward_history.append(score)
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])
            average_reward_history.append(avg_score)

            if avg_score > best_score:
                best_score = avg_score
                agent.save_models()

            print('episode ', i, 'score %.1f' % score,
                    'average score %.1f' % avg_score, flush=True)

        agent.save_models("_last")
        x = [i+1 for i in range(n_games)]
        plc_ddpg(x, score_history, figure_file)

        with open(step_rewards_file, 'w') as f:
            write = csv.writer(f)
            write.writerow(step_reward_history)

        with open(episode_rewards_file, 'w') as f:
            write = csv.writer(f)
            write.writerow(episode_reward_history)

        with open(average_rewards_file, 'w') as f:
            write = csv.writer(f)
            write.writerow(average_reward_history)

        os.kill(pid, signal.SIGINT)
        print("kill signal sent!", flush=True)
        exit()

#------------------------------------------------------------------------------------------------------------

def graph_state():


    rospy.init_node('graph_state' + str(id))
    q_size = 10


    def visu_cb(msg):
        global time_list
        global pos_x_list
        global pos_y_list
        global pos_z_list
        global ori_x_list
        global ori_y_list
        global ori_z_list

        time_list.append(msg.data[0])
        pos_x_list.append(msg.data[1])
        pos_y_list.append(msg.data[2])
        pos_z_list.append(msg.data[3])
        ori_x_list.append(msg.data[4])
        ori_y_list.append(msg.data[5])
        ori_z_list.append(msg.data[6])

    plt.ion() 
    plt.tight_layout(pad = 5) #TODO

    # Create figure
    fig = plt.figure(figsize=(12, 12))

    # Define grid spec
    gs = fig.add_gridspec(3, 3)

    # Create subplots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    ax7 = fig.add_subplot(gs[2, :])

    # Initialize data
    x = [0]
    y1 = [0]
    y2 = [0]
    y3 = [0]
    y4 = [0]
    y5 = [0]
    y6 = [0]
    y7 = [0]

    # Create initial plots
    lines = []
    lines.append(ax1.plot(x, y1, color='b')[0])
    lines.append(ax2.plot(x, y2, color='r')[0])
    lines.append(ax3.plot(x, y3, color='g')[0])
    lines.append(ax4.plot(x, y4, color='m')[0])
    lines.append(ax5.plot(x, y5, color='c')[0])
    lines.append(ax6.plot(x, y6, color='y')[0])
    lines.append(ax7.plot(x, y7, color='k')[0])

    # Set titles for subplots
    ax1.set_title('X position')
    ax2.set_title('Y position')
    ax3.set_title('Z position')
    ax4.set_title('X rotation')
    ax5.set_title('Y rotation')
    ax6.set_title('Z rotation')
    ax7.set_title('Reward')

    # Set labels for subplots
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
        ax.set_xlabel('x')
        ax.set_ylabel('time')

    rospy.Subscriber('/visu', Float32MultiArray, visu_cb, queue_size = q_size)

    graph_padding = 0.1

    rate = rospy.Rate(10)
    counter = 0
    while not rospy.is_shutdown():
        global time_list
        global pos_x_list
        global pos_y_list
        global pos_z_list
        global ori_x_list
        global ori_y_list
        global ori_z_list

        pos_min = min([min(pos_x_list), min(pos_y_list), min(pos_z_list)])
        pos_max = max([max(pos_x_list), max(pos_y_list), max(pos_z_list)])

        padded_pos_min = pos_min - graph_padding
        padded_pos_max = pos_max + graph_padding

        ori_min = min([min(ori_x_list), min(ori_y_list), min(ori_z_list)])
        ori_max = max([max(ori_x_list), max(ori_y_list), max(ori_z_list)])

        padded_ori_min = ori_min - graph_padding
        padded_ori_max = ori_max + graph_padding

        ax1.set_ylim(padded_pos_min, padded_pos_max)
        ax1.set_xlim(-0.1, time_list[-1])

        ax2.set_ylim(padded_pos_min, padded_pos_max)
        ax2.set_xlim(-0.1, time_list[-1])

        ax3.set_ylim(padded_pos_min, padded_pos_max)
        ax3.set_xlim(-0.1, time_list[-1])

        ax4.set_ylim(padded_ori_min, padded_ori_max)
        ax4.set_xlim(-0.1, time_list[-1])

        ax5.set_ylim(padded_ori_min, padded_ori_max)
        ax5.set_xlim(-0.1, time_list[-1])

        ax6.set_ylim(padded_ori_min, padded_ori_max)
        ax6.set_xlim(-0.1, time_list[-1])

        lines[0].set_ydata(pos_x_list)
        lines[1].set_ydata(pos_y_list)
        lines[2].set_ydata(pos_z_list)
        lines[3].set_ydata(ori_x_list)
        lines[4].set_ydata(ori_y_list)
        lines[5].set_ydata(ori_z_list)
        lines[6].set_ydata(pos_x_list) #TODO

        for line in lines:
            line.set_xdata(time_list)

        fig.canvas.draw() 
        fig.canvas.flush_events() 
        rate.sleep()

#------------------------------------------------------------------------------------------------------------

def graph_windowed_reward(q):

    step_counter = 0

    #interactive mode on
    plt.ion() 
    plt.tight_layout(pad = 5) #TODO

    # Create figure
    fig = plt.figure(figsize=(12, 12))

    # Define grid spec
    gs = fig.add_gridspec(4, 3)

    # Create subplots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    ax7 = fig.add_subplot(gs[2, 1])
    ax8 = fig.add_subplot(gs[3, :])

    reward_values = q.get()

    crw = reward_values #copy of reward_values

    mykeys = list(reward_values.keys())

    # Initialize data
    x = [step_counter]
    y1 = crw[mykeys[0]]
    y2 = crw[mykeys[1]]
    y3 = crw[mykeys[2]]
    y4 = crw[mykeys[3]]
    y5 = crw[mykeys[4]]
    y6 = crw[mykeys[5]]
    y7 = crw[mykeys[6]]
    y8 = crw[mykeys[7]]

    # Create initial plots
    lines = []
    lines.append(ax1.plot(x, y1, color='b')[0])
    lines.append(ax2.plot(x, y2, color='r')[0])
    lines.append(ax3.plot(x, y3, color='g')[0])
    lines.append(ax4.plot(x, y4, color='m')[0])
    lines.append(ax5.plot(x, y5, color='c')[0])
    lines.append(ax6.plot(x, y6, color='y')[0])
    lines.append(ax7.plot(x, y7, color='k')[0])
    lines.append(ax8.plot(x, y8, color='k')[0])

    

    # Set titles for subplots
    ax1.set_title(mykeys[0])
    ax2.set_title(mykeys[1])
    ax3.set_title(mykeys[2])
    ax4.set_title(mykeys[3])
    ax5.set_title(mykeys[4])
    ax6.set_title(mykeys[5])
    ax7.set_title(mykeys[6])
    ax8.set_title(mykeys[7])
    # Set labels for subplots
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
        ax.set_xlabel('x')
        ax.set_ylabel('time')

    graph_padding = 0.1

    counter = 0

    step_list = []

    step_list.append(counter)

    while True:
        counter += 1
        step_list.append(counter)

        reward_values = q.get()

        for key in list(reward_values.keys()):
            crw[key].append(reward_values[key][0])

        first_min = min([min(crw[mykeys[0]]), min(crw[mykeys[1]]), min(crw[mykeys[2]])])
        first_max = max([max(crw[mykeys[0]]), max(crw[mykeys[1]]), max(crw[mykeys[2]])])

        padded_first_min = first_min - graph_padding
        padded_first_max = first_max + graph_padding

        second_min = min([min(crw[mykeys[3]]), min(crw[mykeys[4]]), min(crw[mykeys[5]])])
        second_max = max([max(crw[mykeys[3]]), max(crw[mykeys[4]]), max(crw[mykeys[5]])])

        padded_second_min = second_min - graph_padding
        padded_second_max = second_max + graph_padding

        ax1.set_ylim(padded_first_min, padded_first_max)
        ax1.set_xlim(-0.1, step_list[-1])

        ax2.set_ylim(padded_first_min, padded_first_max)
        ax2.set_xlim(-0.1, step_list[-1])

        ax3.set_ylim(padded_first_min, padded_first_max)
        ax3.set_xlim(-0.1, step_list[-1])

        ax4.set_ylim(padded_second_min, padded_second_max)
        ax4.set_xlim(-0.1, step_list[-1])

        ax5.set_ylim(padded_second_min, padded_second_max)
        ax5.set_xlim(-0.1, step_list[-1])

        ax6.set_ylim(padded_second_min, padded_second_max)
        ax6.set_xlim(-0.1, step_list[-1])

        ax7.set_ylim(min(crw[mykeys[6]]), max(crw[mykeys[6]]))
        ax7.set_xlim(-0.1, step_list[-1])

        ax8.set_ylim(min(crw[mykeys[7]]), max(crw[mykeys[7]]))
        ax8.set_xlim(-0.1, step_list[-1])


        lines[0].set_ydata(crw[mykeys[0]])
        lines[1].set_ydata(crw[mykeys[1]])
        lines[2].set_ydata(crw[mykeys[2]])
        lines[3].set_ydata(crw[mykeys[3]])
        lines[4].set_ydata(crw[mykeys[4]])
        lines[5].set_ydata(crw[mykeys[5]])
        lines[6].set_ydata(crw[mykeys[6]])
        lines[7].set_ydata(crw[mykeys[7]])

        for line in lines:
            line.set_xdata(step_list)

        fig.canvas.draw() 
        fig.canvas.flush_events() 

#------------------------------------------------------------------------------------------------------------

def graph_current_reward(q):

    print("mydebug - graph_current_reward started", flush=True)

    myflag = 0

    def sigalrm_handler(*args):
        nonlocal myflag
        myflag = 1


    signal.signal(signal.SIGALRM, sigalrm_handler)
    print("mydebug - Signal handler is set up!!!!!", flush=True)
    

    step_counter = 0

    #interactive mode on
    plt.ion() 
    plt.tight_layout(pad = 5) #TODO

    # Create figure
    fig = plt.figure(figsize=(12, 12))

    # Define grid spec
    gs = fig.add_gridspec(4, 3)

    # Create subplots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    ax7 = fig.add_subplot(gs[2, 1])
    ax8 = fig.add_subplot(gs[3, :])

    reward_values = q.get()

    crw = reward_values #copy of reward_values

    mykeys = list(reward_values.keys())

    # Initialize data
    x = [step_counter]
    y1 = crw[mykeys[0]]
    y2 = crw[mykeys[1]]
    y3 = crw[mykeys[2]]
    y4 = crw[mykeys[3]]
    y5 = crw[mykeys[4]]
    y6 = crw[mykeys[5]]
    y7 = crw[mykeys[6]]
    y8 = crw[mykeys[7]]

    # Create initial plots
    lines = []
    lines.append(ax1.plot(x, y1, color='b')[0])
    lines.append(ax2.plot(x, y2, color='r')[0])
    lines.append(ax3.plot(x, y3, color='g')[0])
    lines.append(ax4.plot(x, y4, color='m')[0])
    lines.append(ax5.plot(x, y5, color='c')[0])
    lines.append(ax6.plot(x, y6, color='y')[0])
    lines.append(ax7.plot(x, y7, color='k')[0])
    lines.append(ax8.plot(x, y8, color='k')[0])

    

    # Set titles for subplots
    ax1.set_title(mykeys[0])
    ax2.set_title(mykeys[1])
    ax3.set_title(mykeys[2])
    ax4.set_title(mykeys[3])
    ax5.set_title(mykeys[4])
    ax6.set_title(mykeys[5])
    ax7.set_title(mykeys[6])
    ax8.set_title(mykeys[7])
    # Set labels for subplots
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
        ax.set_xlabel('x')
        ax.set_ylabel('time')

    graph_padding = 0.1

    counter = 0

    step_list = []

    step_list.append(counter)

    skip = 0
    while True:
        bla = 0
        if myflag == 1:
            
            for key in list(crw.keys()):
                crw[key]=[]
            counter = -1
            step_list = []
            myflag = 0
            bla = 1


        reward_values = q.get()

        if bla:
            print("first values after fall: ", reward_values, flush=True)

        counter += 1
        step_list.append(counter)

        for key in list(reward_values.keys()):
            crw[key].append(reward_values[key][0])


        graph_padding = 1.5
        
        first_min = min([min(crw[mykeys[0]]), min(crw[mykeys[1]]), min(crw[mykeys[2]])])
        first_max = max([max(crw[mykeys[0]]), max(crw[mykeys[1]]), max(crw[mykeys[2]])])

        padded_first_min = first_min - graph_padding
        padded_first_max = first_max + graph_padding

        second_min = min([min(crw[mykeys[3]]), min(crw[mykeys[4]]), min(crw[mykeys[5]])])
        second_max = max([max(crw[mykeys[3]]), max(crw[mykeys[4]]), max(crw[mykeys[5]])])

        padded_second_min = second_min - graph_padding
        padded_second_max = second_max + graph_padding

        ax1.set_ylim(padded_first_min, padded_first_max)
        ax1.set_xlim(-0.1, step_list[-1])

        ax2.set_ylim(padded_first_min, padded_first_max)
        ax2.set_xlim(-0.1, step_list[-1])

        ax3.set_ylim(padded_first_min, padded_first_max)
        ax3.set_xlim(-0.1, step_list[-1])

        ax4.set_ylim(padded_second_min, padded_second_max)
        ax4.set_xlim(-0.1, step_list[-1])

        ax5.set_ylim(padded_second_min, padded_second_max)
        ax5.set_xlim(-0.1, step_list[-1])

        ax6.set_ylim(padded_second_min, padded_second_max)
        ax6.set_xlim(-0.1, step_list[-1])

        third_min = min(crw[mykeys[6]]) - graph_padding
        third_max = max(crw[mykeys[6]]) + graph_padding

        ax7.set_ylim(third_min, third_max)
        ax7.set_xlim(-0.1, step_list[-1])

        fourth_min = min(crw[mykeys[7]]) - graph_padding
        fourth_max = max(crw[mykeys[7]]) + graph_padding

        ax8.set_ylim(fourth_min, fourth_max)
        ax8.set_xlim(-0.1, step_list[-1])

        lines[0].set_ydata(crw[mykeys[0]])
        lines[1].set_ydata(crw[mykeys[1]])
        lines[2].set_ydata(crw[mykeys[2]])
        lines[3].set_ydata(crw[mykeys[3]])
        lines[4].set_ydata(crw[mykeys[4]])
        lines[5].set_ydata(crw[mykeys[5]])
        lines[6].set_ydata(crw[mykeys[6]])
        lines[7].set_ydata(crw[mykeys[7]])

        for line in lines:
            line.set_xdata(step_list)

        fig.canvas.draw() 
        fig.canvas.flush_events() 

#------------------------------------------------------------------------------------------------------------


def main_process(
        pid, 
        interface_output_file_path, 
        controller_output_file_path, 
        controller_output_folder_path,  
        id
        ):
    print("Interface process with id = ", id, "created!", flush=True)
    with open(interface_output_file_path, 'w') as f:
        sys.stdout = f
        first = 1


        def shift_elements(lst, new_elements, reg_len, shifting_amount):
            """
            Shifts three new elements into a list of length 18, initially filled with zeros.
            
            Parameters:
            lst (list): The original list of length 18.
            new_elements (list): A list of three new elements to be added.
            
            Returns:
            list: The updated list after shifting in the new elements.
            """
            if len(lst) != reg_len:
                raise ValueError("The original list must be of length " + str(reg_len))
            if len(new_elements) != shifting_amount:
                raise ValueError("The new elements list must contain exactly " + str(shifting_amount) +  " elements but got " + str(len(new_elements)))
        
            # Remove the last three elements
            lst = lst[:-shifting_amount]
            # Add the new elements to the front
            lst = new_elements + lst
            return lst

        def map_to_discrete_range(value):
            input_min, input_max = -1, 1
            output_min, output_max = -1, 1

            # Clip the value to the input range
            value = np.clip(value, input_min, input_max)
            
            # Normalize the value to a 0-1 range
            normalized_value = (value - input_min) / (input_max - input_min)
            
            # Scale to the output range and round to the nearest integer
            discrete_value = np.round(normalized_value * (output_max - output_min) + output_min).astype(float)
            
            return discrete_value

        def sigint_handler(*args):
            print("\n exiting!!!", flush=True)
            stop_publisher.publish(Bool(True))

            for p in processes:
                p.kill()
                p.join()

            print("Processes should be joined by now", flush=True)
            os.kill(pid, signal.SIGALRM)
            exit(0)

        def sigquit_handler(*args):
            print("You pressed Ctrl + \\", flush=True)

        def sigalrm_handler(*args):
            global step_cb_enable
            global cold_start
            global first

            # os.kill(os.getpid(p4), signal.SIGALRM)

            cold_start = True
            # print("sigalarm received in main form subprocess", flush = True)
            z = Bool(True)

            delay = 0.3

            step_cb_enable = False

            sync_publisher.publish(z)   #synchronize
            # print("sync publisher called", flush=True)
            time.sleep(delay)

            stop_publisher.publish(z)  #stop simulation
            # print("stop publisher called", flush=True)
            time.sleep(delay)

            start_publisher.publish(z)  #start simulation
            # print("start publisher called", flush=True)
            time.sleep(delay)

            step_publisher.publish(z)   #next step
            # print("trig ", flush=True)
            time.sleep(delay)

            step_cb_enable = True

            step_publisher.publish(z) #needed because the simulator doesnt publish the states with only one step
            # print("trig next2", flush=True)
            time.sleep(delay)

            # print("sigalarm handling ended\n", flush = True)

            return
            # step_cb("alma")

        def state_cb(msg):
            # if state_cb_enable 
            # print("state callback called!!!", flush=True)

            global actual_joint_positions
            global actual_pos
            global actual_ori
            global actual_time
            global got_state

            data_list = list(msg.data)
            actual_time = data_list[0]
            actual_pos = data_list[1:4]
            actual_ori = data_list[4:7]
            actual_joint_positions = data_list[7:]

            got_state = True
            return 

        def simState_cb(msg):
            # print("simstate callback called!!!!", flush=True)

            global sim_state
            global got_simstate
            sim_state = msg
            got_simstate = True

            return
        
        def is_it_done(state):
            return 0
        
        def is_fallen(pos):
            global sim_state
            if pos[2] < 0.4:
                return True
            return False
        
        def reward_function(actual_time, actual_pos, actual_ori, actual_joint_positions, speed, ang_speed):

            global original_pos
            global original_ori

           


            def radians_to_degrees(radian_list):
                return [radian * (180 / math.pi) for radian in radian_list]



            def get_limited_joints(joint_pos):

                buffer = 2

                limits = [ \
                    [-45, 45],
                    [-45, 45],
                    [-90, 90],
                    [-45, 135],
                    [-45, 45],
                    [-45, 45],
                    #
                    [-45, 45],
                    [-45, 45],
                    [-90, 90],
                    [-45, 135],
                    [-45, 45],
                    [-45, 45],
                    #
                    [-90, 90],
                    [-45, 45],
                    #
                    [-180, 90],
                    [-15, 90],
                    [-45, 45],
                    #
                    [-180, 90],
                    [-15, 90],
                    [-45, 45],
                ]

                counter = 0
                joints_limited = []

                for i in range(len(joint_pos)):
                    if (joint_pos[i] <= (limits[i][0] + buffer)) or \
                        (joint_pos[i] >= (limits[i][1] - buffer)): 
                        counter += 1
                        joints_limited.append(1)

                    else:
                        joints_limited.append(0)

                return counter, joints_limited

            oxp = original_pos[0]
            oyp = original_pos[1]
            ozp = original_pos[2]

            oxo = original_ori[0]
            oyo = original_ori[1]
            ozo = original_ori[2]

            axp = actual_pos[0]
            ayp = actual_pos[1]
            azp = actual_pos[2]

            axo = actual_ori[0]
            ayo = actual_ori[1]
            azo = actual_ori[2]


            forward_weight = 60
            lateral_weigth = 10
            vertical_weigth = 5

            x_rot_weight = 3
            y_rot_weight = 10
            z_rot_weight = 5

            limits_weight = 0.3
            fall_weight = 0
            time_weight = 15

            speed_x_weight = 2
            speed_y_weight = 20
            speed_z_weight = 20
            ang_speed_x_weight = 20
            ang_speed_y_weight = 20
            ang_speed_z_weight = 20
            
            if azp < 0.4:
                fall_weight = 500


            fall_reward = 1

            #megtett tav
            forward_reward = abs(oxp) + axp
            lateral_reward = abs(abs(oyp)-abs(ayp))
            vertical_reward = abs(abs(ozp)-abs(azp))

            x_rot_reward = abs(abs(oxo) - abs(axo))
            y_rot_reward = abs(abs(oyo) - abs(ayo))
            z_rot_reward = abs(abs(ozo) - abs(azo))


            deg_joint_pos = radians_to_degrees(actual_joint_positions)
            limits_reward, joints_limited = get_limited_joints(deg_joint_pos)



            forward = forward_weight            * forward_reward    
            lateral = lateral_weigth            * lateral_reward    * -1
            vertical = vertical_weigth          * vertical_reward   * -1
            x_rot = x_rot_weight                * x_rot_reward      * -1
            y_rot = y_rot_weight                * y_rot_reward      * -1
            z_rot = z_rot_weight                * z_rot_reward      * -1
            limits = limits_weight              * limits_reward     * -1
            fall = fall_weight                  * fall_reward       * -1
            time = time_weight                  * actual_time       * +1
            # speed_x = speed_x_weight          * speed[0]          * +1
            speed_y = speed_y_weight            * speed[1]          * -1
            speed_z = speed_z_weight            * speed[2]          * -1
            ang_speed_x = ang_speed_x_weight    * ang_speed[0]      * -1
            ang_speed_y = ang_speed_y_weight    * ang_speed[1]      * -1
            ang_speed_z = ang_speed_z_weight    * ang_speed[2]      * -1


            

            reward = 0      \
            + forward       \
            + lateral       \
            + vertical      \
            + x_rot         \
            + y_rot         \
            + z_rot         \
            + limits        \
            + fall          \
            + time          \
            + speed_y       \
            + speed_z       \
            + ang_speed_x   \
            + ang_speed_y   \
            + ang_speed_z



            # print('\n')
            # print("forward", forward, flush=True)
            # print("lateral", lateral, flush=True)
            # print("vertical", vertical, flush=True)
            # print("x_rot", x_rot, flush=True)
            # print("y_rot", y_rot, flush=True)
            # print("z_rot", z_rot, flush=True)
            # print("limits", limits, flush=True)
            # print("fall", fall, flush=True)
            # print("time", time, flush=True)
            # print("speed_y", speed_y, flush=True)
            # print("speed_z", speed_z, flush=True)
            # print("ang_speed_x", ang_speed_x, flush=True)
            # print("ang_speed_y", ang_speed_y, flush=True)
            # print("ang_speed_z", ang_speed_z, flush=True)
            # print("reward", reward)
            # print('\n')


            

            reward_values = \
            {
                'forward'   : [forward],
                'lateral'   : [lateral],
                'vertical'  : [vertical],
                'x_rot'     : [x_rot],
                'y_rot'     : [y_rot],
                'z_rot'     : [z_rot],
                'limits'    : [limits],
                'reward'    : [reward]
            }


            return reward/10, reward_values
        
        def calc_core_velocities(actual_pos, actual_ori):
            global prev_pos
            global prev_ori
            global dt

            # diff_pos = abs(prev_pos - actual_pos)#the sign matters for the first element!!!!
            # diff_ori = abs(prev_ori - actual_ori) #the sign doesnt matter, any difference is bad

            diff_pos = [abs(a - b) for a, b in zip(prev_pos, actual_pos)]
            diff_ori = [abs(a - b) for a, b in zip(prev_ori, actual_ori)]

            prev_pos = actual_pos
            prev_ori = actual_ori

            # speed = diff_pos / dt
            speed = [a/dt for a in diff_pos]
            # ang_speed = diff_ori / dt
            ang_speed = [a/dt for a in diff_ori]

            return speed, ang_speed
        
        def calc_joint_velocities(actual_joint_positions):
            global prev_joint_positions

            # joint_velocities = prev_joint_positions - actual_joint_positions #its a state value so we don't alter it
            joint_velocities = [abs(a - b) for a, b in zip(prev_joint_positions, actual_joint_positions)]
            prev_joint_positions = actual_joint_positions #update the previous value for next calculation
            return joint_velocities

        def step_cb(msg):
            global step_cb_enable

            global got_state

            global actual_joint_positions
            global actual_pos
            global actual_ori

            global prev_pos
            global prev_ori
            global prev_joint_positions

            global actual_time

            global m2s
            global s2m

            global pos_shift_reg
            global ori_shift_reg
            global joint_positions_shift_reg

            global original_pos
            global original_ori

            global cold_start

            global joint_velocities_shift_reg
            global core_speed_shift_reg
            global core_ang_speed_shift_reg
            global dt
    

            # print("mydebug - step callback called", flush=True)


            if step_cb_enable: # if the data is correct
                # print("mydebug - enabled - step callback called!!!", flush=True)

                #wait until all the state variables are known
                while not got_state:
                    time.sleep(0.001)
                    # print("waiting for state, now it's ", got_state, flush=True)
                got_state = False

                pos_shift_reg = shift_elements(pos_shift_reg, actual_pos, 9, 3) #TODO generalize
                ori_shift_reg = shift_elements(ori_shift_reg, actual_ori, 9, 3) #TODO generalize
                joint_positions_shift_reg = shift_elements(joint_positions_shift_reg, actual_joint_positions, 60, 20)

                if cold_start:
                    #velocities are 0 at the start
                    speed = [0] * 3
                    ang_speed = [0] * 3
                    joint_velocities = [0] * 20
                else:
                    speed, ang_speed = calc_core_velocities(actual_pos, actual_ori) #needs previous value for calculation
                    joint_velocities = calc_joint_velocities(actual_joint_positions)  #needs previous value for calculation

                core_speed_shift_reg = shift_elements(core_speed_shift_reg, speed, 9, 3) 
                core_ang_speed_shift_reg = shift_elements(core_ang_speed_shift_reg, speed, 9, 3) 
                    

                joint_velocities_shift_reg = shift_elements(joint_velocities_shift_reg, joint_velocities, 60, 20)

                state = [*pos_shift_reg, *ori_shift_reg, *joint_positions_shift_reg, *core_speed_shift_reg, *core_ang_speed_shift_reg, *joint_velocities_shift_reg]
                # 9 9 60 9 9 60 = 156


                #at this point we have the state!!!!!!!!
                if cold_start:
                    cold_start = False

                    #original position needs to be saved for later reference
                    original_pos = actual_pos
                    original_ori = actual_ori

                    #in first call we save the state as previous state
                    prev_pos = actual_pos
                    prev_ori = actual_ori
                    prev_joint_positions = actual_joint_positions


                    m2s.put(state)
                    action = s2m.get()

                    mapped_action = action[0].tolist()
                    mapped_action = [x / 0.05 for x in mapped_action]

                    action_packet = Float32MultiArray()
                    action_packet.data = mapped_action
                
                    joint_publisher0.publish(action_packet)
                    step_publisher.publish(Bool(True))

                    # dt = actual_time
                    # print("dt = ", dt, flush=True)
                    dt = 0.05
                
                    return

                else:

                    m2s.put(state)

                    reward, reward_values = reward_function(actual_time, actual_pos, actual_ori, actual_joint_positions, speed, ang_speed)
                    # q4.put(reward_values)
                    m2s.put(reward)

                    achieved = is_it_done(actual_pos)
                    fallen = is_fallen(actual_pos)
                    done = fallen or achieved
                    m2s.put(done)
                    if done:
                        return

                    action = s2m.get()

                mapped_action = action[0].tolist()
                mapped_action = [x / 0.05 for x in mapped_action]

                action_packet = Float32MultiArray()
                action_packet.data = mapped_action
            
                joint_publisher0.publish(action_packet)
                step_publisher.publish(Bool(True))
                # print("step publisher called\n", flush=True)
                
                return
            else:
                # print("mydebug - disabled - step callback called!!!", flush=True)

                return
        


        global pos_shift_reg 
        pos_shift_reg = [0] * 9

        global ori_shift_reg
        ori_shift_reg = [0] * 9

        global joint_positions_shift_reg
        joint_positions_shift_reg = [0] * 60

        global joint_velocities_shift_reg
        joint_velocities_shift_reg = [0] * 60

        global core_speed_shift_reg
        core_speed_shift_reg = [0] * 9

        global core_ang_speed_shift_reg
        core_ang_speed_shift_reg = [0] * 9


        pause_flag = True
        sim_state = Float32MultiArray()

        processes = []

        # mp.set_start_method('spawn')
        global m2s
        m2s = mp.Queue()
        global s2m
        s2m = mp.Queue()

        # p1 = mp.Process(target=joint_control_ddpg, args=(m2s, s2m), daemon=True)
        # p1 = mp.Process(target=joint_control_ddpg, args=(m2s, s2m, os.getpid(), file_path, id), daemon=True)
        p1 = mp.Process(
            target=joint_control_ddpg, 
            args=(
                m2s, 
                s2m, 
                os.getpid(), 
                controller_output_file_path, 
                controller_output_folder_path, 
                id), 
            daemon=True)
        processes.append(p1)
        # p1 = Process(target=joint_control, args=(q1,))
        p1.start()

        # q2 = Queue()
        # p2 = Process(target=process_image, args=(q2,))
        #processes.append(p2)
        # p2.start()

        # q3 = Queue()
        # p3 = Process(target=graph_state, args=[])
        # processes.append(p3)
        # p3.start()

        # q4 = Queue()
        # p4 = Process(target=graph_current_reward, args=[q4,])
        # processes.append(p4)
        # p4.start()

        # p5 = Process(target=graph_windowed_reward, args=[q4,])
        # processes.append(p5)
        # p5.start()
        
        rospy.init_node('hugo_main' + str(id))
        q_size = 1

        sync_publisher = rospy.Publisher("/enableSyncMode" + str(id), Bool, queue_size=q_size)#, latch=True)
        start_publisher = rospy.Publisher("/startSimulation" + str(id), Bool, queue_size=q_size)#, latch=True)
        stop_publisher = rospy.Publisher("/stopSimulation" + str(id), Bool, queue_size=q_size)#, latch=True)
        step_publisher = rospy.Publisher("/triggerNextStep" + str(id), Bool, queue_size=q_size)#, latch=True)
        puse_publisher = rospy.Publisher("/pauseSimulation" + str(id), Bool, queue_size=q_size)
        joint_publisher0 = rospy.Publisher('/action' + str(id), Float32MultiArray, queue_size=q_size)

        rospy.Subscriber("/simulationState" + str(id), Int32, simState_cb)
        rospy.Subscriber("/state" + str(id), Float32MultiArray, state_cb, queue_size = q_size)
        rospy.Subscriber("/simulationStepDone" + str(id), Bool, step_cb, queue_size = q_size)#, latch=True)
        
        time.sleep(0.1) #original value 5
        
        rate = rospy.Rate(10)

        time.sleep(0.5)
        print("initialization done", flush=True)

        signal.signal(signal.SIGINT, sigint_handler)
        signal.signal(signal.SIGQUIT, sigquit_handler)
        signal.signal(signal.SIGALRM, sigalrm_handler)

        print("main thread", flush=True)

        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            rate.sleep()





def create_next_run_folder(path):
    # Regular expression to match "run" followed by a number
    pattern = re.compile(r'^run(\d+)$')
    
    highest_number = -1
    
    # Check if the path exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified path does not exist: {path}")
    
    # Iterate over the items in the directory
    for dir_name in os.listdir(path):
        # Check if the directory name matches the pattern
        match = pattern.match(dir_name)
        if match:
            # Extract the number and convert to integer
            number = int(match.group(1))
            # Update the highest number if necessary
            if number > highest_number:
                highest_number = number
    
    # Increment the highest number by one for the new folder name
    next_number = highest_number + 1
    new_folder_name = f"run{next_number}"
    new_folder_path = os.path.join(path, new_folder_name)
    
    # Create the new folder
    os.makedirs(new_folder_path)
    os.makedirs(new_folder_path + "/simulator")
    os.makedirs(new_folder_path + "/interface")
    os.makedirs(new_folder_path + "/controller")
    os.makedirs(new_folder_path + "/controller/plots")
    os.makedirs(new_folder_path + "/controller/rewards/step_rewards")
    os.makedirs(new_folder_path + "/controller/rewards/episode_rewards")
    os.makedirs(new_folder_path + "/controller/rewards/average_rewards")
    os.makedirs(new_folder_path + "/controller/checkpoints")
    os.makedirs(new_folder_path + "/controller/checkpoints/last")
    os.makedirs(new_folder_path + "/controller/checkpoints/best")
    
    print("\nOutput folder: " + new_folder_path + " created", flush=True)
    return new_folder_path



if __name__ == '__main__':

    counter = 0

    mp.set_start_method('spawn')


    def sigint_handler(*args):
        print("Shutting down the processes", flush=True)
        time.sleep(1)

        for process, filename in simulator_processes:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            # os.kill(process.pid, signal.SIGTERM)
            # print("pid", process.pid)
            # process.kill()
            
            # process.terminate()  # Terminate the process
            process.wait()       # Wait for the process to terminate
            if process.returncode is not None:
                print(f"Process writing to {filename} terminated with return code {process.returncode}")


        for process in main_processes:
            os.kill(process.pid, signal.SIGINT)


        print("All processes have been terminated.")
        exit(0)

    def sigalrm_handler(*args):
        global counter
        # print("sigalarm received", flush=True)

        counter += 1
        if counter == num_instances:

            for process, filename in simulator_processes:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                # os.kill(process.pid, signal.SIGTERM)
                # print("pid", process.pid)
                # process.kill()
                
                # process.terminate()  # Terminate the process
                process.wait()       # Wait for the process to terminate
                if process.returncode is not None:
                    print(f"Process writing to {filename} terminated with return code {process.returncode}")

            exit()


    signal.signal(signal.SIGINT, sigint_handler)
    signal.signal(signal.SIGALRM, sigalrm_handler)

    #-------------------------------------------------------------------------------------

    num_instances = 10

    bash_commands = []
    for i in range(num_instances):
        command = "/home/kovacs/Downloads/CoppeliaSim_Edu_V4_6_0_rev18_Ubuntu20_04/coppeliaSim.sh -h -gparam1=" + str(i) + " -GROSInterface.nodeName=MyNodeName" + str(i) + " /home/kovacs/Documents/disszertacio/hugo_python_control_coppeliasim_v4/asti.ttt"
        bash_commands.append(command)

    path = "/home/kovacs/Documents/disszertacio/hugo_python_control_coppeliasim_v4/results"
    base_folder = create_next_run_folder(path)


    print("\nStarting simulator processes", flush=True)
    simulator_processes = []
    for i, command in enumerate(bash_commands):
        with open(base_folder + f"/simulator/sim_out_{i}.txt", "w") as outfile:
            arguments = command.split()
            process = subprocess.Popen([arguments[0]] + arguments[1:], shell=False, stdout=outfile, stderr=outfile, preexec_fn=os.setsid)
            simulator_processes.append((process, f"/simulator/sim_out_{i}.txt"))

    print("Simulators started", flush=True)
    time.sleep(7)


    print("\nStarting controller processes", flush=True)
    main_processes = []
    for idx in range(num_instances):
        interface_output_file_path      = base_folder + "/interface/int_out_" + str(idx) +".txt"
        controller_output_file_path     = base_folder +  "/controller/cont_out_" + str(idx) +".txt"
        controller_output_folder_path   = base_folder +  "/controller/"
        p = mp.Process(target=main_process, args=
                       (
                        os.getpid(), 
                        interface_output_file_path, 
                        controller_output_file_path, 
                        controller_output_folder_path, 
                        idx
                        ))
        main_processes.append(p)
        p.start()
    print("Controllers started", flush=True)

    while True:
        time.sleep(1)

