#!/usr/bin/env python3

import torch.multiprocessing.queue
import rospy
from sensor_msgs.msg import Image # ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError # OpenCV2 for saving an image
import cv2
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
# import torch.multiprocessing.queue as queue
# from multiprocessing import Manager


got_state = False


counter = 0
prev_state = 0

original_pos = []
original_ori = []


logger = logging.getLogger('train')
logger.setLevel(logging.DEBUG)

# Instantiate CvBridge
bridge = CvBridge()

# position_cb_enable = True
# orientation_cb_enable = True
# joint_positions_cb_enable = True

got_position = False
# got_orientiation = False
# got_joint_positions = False
got_simstate = False

# got_position = True
# got_orientiation = True
# got_joint_positions = True

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
# actual_joint_positions = [0] * 20 



# pos_x = 0
# pos_y = 0
# pos_z = 0
# ori_x = 0
# ori_y = 0
# ori_z = 0




#------------------------------------------------------------------------------------------------------------

def joint_control(m2s, s2m):
    print("torch process started!!!", flush=True)
    # rospy.init_node('hugo_control')
    # q_size = 10
    # rospy.Subscriber("/simulationStepDone", Bool, step_cb, queue_size = q_size)#, latch=True)


    writer = SummaryWriter('runs/run_1')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")


    logger.info("Using {}".format(device))
    checkpoint_dir = '.'
    seed = 42

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    gamma = 0.99
    tau = default=0.001
    hidden_size = [400, 300]
    observation_space_dim = 96
    action_space_dim = 26
    replay_size = 1e6
    noise_stddev = 0.2
    load_model = False
    timesteps = 1e6
    n_test_cycles = 10
    batch_size = 1
    done = 0
    reward_threshold = 1

    agent = DDPG(gamma,
                tau,
                hidden_size,
                observation_space_dim,
                action_space_dim,
                checkpoint_dir=checkpoint_dir
                )
    memory = ReplayMemory(int(replay_size))

    nb_actions = action_space_dim
    ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions),
                                            sigma=float(noise_stddev) * np.ones(nb_actions))




    # Define counters and other variables
    start_step = 0
    # timestep = start_step
    if load_model:
        # Load agent if necessary
        start_step, memory = agent.load_checkpoint()
    timestep = start_step // 10000 + 1
    rewards, policy_losses, value_losses, mean_test_rewards = [], [], [], []
    epoch = 0
    t = 0
    time_last_checkpoint = time.time()

    # Start training
    # logger.info('Train agent on {} env'.format({env.unwrapped.spec.id}))
    logger.info('Doing {} timesteps'.format(timesteps))
    logger.info('Start at timestep {0} with t = {1}'.format(timestep, t))
    logger.info('Start training at {}'.format(time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))




    while timestep <= timesteps:
        ou_noise.reset()
        epoch_return = 0


        #get state first
        #reset the simulator
        #take a step
        # initial_state = 

        #TODO
        #signal that it's first step
        # print("entering waiting", flush=True)
        # print(list(q.queue))
        mystate = m2s.get()
        # print("waiting ended", flush=True)
        state = torch.Tensor(mystate).to(device)
        state = state.unsqueeze(0)

        while True:
            # if args.render_train:
            #     env.render()
            state.squeeze()
            # print(state)
            action = agent.calc_action(state, ou_noise)

            s2m.put(action)

            #get next state
            #get reward
            #get done
            # next_state, reward, done, _ = env.step(action.cpu().numpy()[0]) 
            next_state = m2s.get()
            reward = m2s.get()
            done = m2s.get()

            timestep += 1
            epoch_return += reward

            mask = torch.Tensor([done]).to(device)
            reward = torch.Tensor([reward]).to(device)
            next_state = torch.Tensor([next_state]).to(device)

            memory.push(state, action, mask, next_state, reward)

            state = next_state

            epoch_value_loss = 0
            epoch_policy_loss = 0

            if len(memory) > batch_size:
                transitions = memory.sample(batch_size)
                # Transpose the batch
                # (see http://stackoverflow.com/a/19343/3343043 for detailed explanation).
                batch = Transition(*zip(*transitions))

                # Update actor and critic according to the batch
                value_loss, policy_loss = agent.update_params(batch)

                epoch_value_loss += value_loss
                epoch_policy_loss += policy_loss

            if done:
                break

        rewards.append(epoch_return)
        value_losses.append(epoch_value_loss)
        policy_losses.append(epoch_policy_loss)
        writer.add_scalar('epoch/return', epoch_return, epoch)

        # Test every 10th episode (== 1e4) steps for a number of test_epochs epochs
        if timestep >= 10000 * t:
            t += 1
            test_rewards = []
            for _ in range(n_test_cycles):
                #get state first
                #reset the simulator
                #take a step
                #initial_state = 
                # state = torch.Tensor([env.reset()]).to(device)
                test_reward = 0
                while True:
                    # if args.render_eval:
                    #     env.render()

                    action = agent.calc_action(state)  # Selection without noise

                    #TODO
                    # next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
                    test_reward += reward

                    next_state = torch.Tensor(next_state).to(device)

                    state = next_state
                    if done:
                        break
                test_rewards.append(test_reward)

            mean_test_rewards.append(np.mean(test_rewards))

            for name, param in agent.actor.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
            for name, param in agent.critic.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

            writer.add_scalar('test/mean_test_return', mean_test_rewards[-1], epoch)
            logger.info("Epoch: {}, current timestep: {}, last reward: {}, "
                        "mean reward: {}, mean test reward {}".format(epoch,
                                                                        timestep,
                                                                        rewards[-1],
                                                                        np.mean(rewards[-10:]),
                                                                        np.mean(test_rewards)))

            # Save if the mean of the last three averaged rewards while testing
            # is greater than the specified reward threshold
            # TODO: Option if no reward threshold is given
            if np.mean(mean_test_rewards[-3:]) >= reward_threshold:
                # agent.save_checkpoint(timestep, memory)
                time_last_checkpoint = time.time()
                # logger.info('Saved model at {}'.format(time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))

        epoch += 1

    # agent.save_checkpoint(timestep, memory)
    logger.info('Saved model at endtime {}'.format(time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))
    logger.info('Stopping training at {}'.format(time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))
    
    #TODO 
    # close the simulator session
    # env.close()

#------------------------------------------------------------------------------------------------------------

def process_image(q):

    def image_cb(msg):
        #print("Received an image!")
        try:
            # Convert your ROS Image message to OpenCV2
            cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
            cv2_img = cv2.flip(cv2_img, 0)
        except CvBridgeError as e:
            print(e)
        else:
            # Save your OpenCV2 image as a jpeg 
            #cv2.imwrite('camera_image.jpeg', cv2_img)
            cv2.namedWindow("Input")
            cv2.imshow("Input", cv2_img)
            cv2.waitKey(1)


    rospy.init_node('hugo_vision')
    q_size = 10
    rospy.Subscriber("/image", Image, image_cb, queue_size=q_size, buff_size=2**10)

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        # print(q.empty())
        # print("hello")
        # if q.empty():
        #     print("Nothing in queue of joint_control")
        # nothing = q.get()
        # print("process image thread process image thread process image thread process image thread")
        rate.sleep()


#------------------------------------------------------------------------------------------------------------


def graph_state():


    rospy.init_node('graph_state')
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




def graph_reward(q):

    print("mydebug - graph_reward started", flush=True)

    def sigalrm_handler(*args):
        print("mydebug - \n\n\n\n!!!!\nSIGALRM RECEIVED\n\n\n", flush=True)
        fig.clf()
    
    signal.signal(signal.SIGALRM, sigalrm_handler)
    print("mydebug - Signal handler is set up!!!!!", flush=True)
    

    step_counter = 0

    #interactive mode on
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
    ax1.set_title(mykeys[0])
    ax2.set_title(mykeys[1])
    ax3.set_title(mykeys[2])
    ax4.set_title(mykeys[3])
    ax5.set_title(mykeys[4])
    ax6.set_title(mykeys[5])
    ax7.set_title(mykeys[6])

    # Set labels for subplots
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
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


        lines[0].set_ydata(crw[mykeys[0]])
        lines[1].set_ydata(crw[mykeys[1]])
        lines[2].set_ydata(crw[mykeys[2]])
        lines[3].set_ydata(crw[mykeys[3]])
        lines[4].set_ydata(crw[mykeys[4]])
        lines[5].set_ydata(crw[mykeys[5]])
        lines[6].set_ydata(crw[mykeys[6]])

        for line in lines:
            line.set_xdata(step_list)

        fig.canvas.draw() 
        fig.canvas.flush_events() 



    

#------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':


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


    #Ctrl-C handling
    def sigint_handler(*args):
        print("\n exiting!!!", flush=True)
        stop_publisher.publish(Bool(True))
        p1.kill()
        p2.kill()
        p1.join()
        p2.join()
        p3.kill()
        p3.join()
        p4.kill()
        p4.join()
        print("Processes should be joined by now", flush=True)
        exit(0)



    def sigquit_handler(*args):
        print("You pressed Ctrl + \\", flush=True)

        # puse_publisher.publish(Bool(pause_flag))
        # pause_flag = not pause_flag


    # def sigstop_handler(*args):
    #     print("You pressed Ctrl + Z")


    # def joint_positions_cb(data):
    #     global actual_joint_positions
    #     global got_joint_positions
    #     global joint_positions_cb_enable

    #     if joint_positions_cb_enable:
    #         actual_joint_positions = list(data.data)
    #         got_joint_positions = True


    def state_cb(msg):
        # if state_cb_enable 
        print("state callback called!!!", flush=True)

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

        # print("\ncheck correctness")
        # print(data_list)
        # print("---")
        # print(actual_time)
        # print(actual_pos)
        # print(actual_ori)
        # print(actual_joint_positions)

        got_state = True
        return 


    # def position_cb(data):
    #     # print("mydebug - position callback called!!!")
    #     if position_cb_enable:
    #         global pos_x
    #         global pos_y
    #         global pos_z
    #         global got_position

    #         pos_x = data.x
    #         pos_y = data.y
    #         pos_z = data.z

    #         got_position = True
    #         # print("Position   : x:", data.x, "   y:", data.y, "   :", data.z)
    


    # def orientation_cb(data):
    #     # print("mydebug - orientation callback called!!!")
    #     if orientation_cb_enable:
    #         global ori_x
    #         global ori_y
    #         global ori_z
    #         global got_orientiation 

    #         ori_x = math.degrees(data.x)
    #         ori_y = math.degrees(data.y)
    #         ori_z = math.degrees(data.z)

    #         ori_x = data.x
    #         ori_y = data.y
    #         ori_z = data.z
            
    #         got_orientiation = True
    #         # print("Orientation   : x:", data.x, "   y:", data.y, "   :", data.z)



    # def simTime_cb(msg):
    #     # print("Simulation time: ",msg)
    #     return



    def simState_cb(msg):
        print("simstate callback called!!!!", flush=True)

        global sim_state
        global got_simstate
        # print("Simulation state: ",msg)
        sim_state = msg
        got_simstate = True

        return
    
    
    def is_it_done(state):
        return 0
    

    def is_fallen(pos):
        global sim_state
        if pos[2] < 0.4:
            # os.kill(p4.pid, signal.SIGALRM)
            return True
        return False
    

    def reward_function(actual_pos, actual_ori):
        global original_pos
        global original_ori

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


        forward_weight = 30
        lateral_weigth = 3
        vertical_weigth = 2
        x_rot_weight = 1.5
        y_rot_weight = 7
        z_rot_weight = 3
        

        #megtett tav
        forward_reward = abs(oxp) + axp
        # print("forward_reward: ", forward_reward)
        # print("oxp: ", oxp)
        # print("axp: ", axp)
        lateral_reward = abs(abs(oyp)-abs(ayp))
        vertical_reward = abs(abs(ozp)-abs(azp))

        x_rot_reward = abs(abs(oxo) - abs(axo))
        y_rot_reward = abs(abs(oyo) - abs(ayo))
        z_rot_reward = abs(abs(ozo) - abs(azo))
    

        # reward = \
        #     forward_weight      * forward_reward \
        # -   lateral_weigth      * lateral_reward \
        # -   vertical_weigth     * vertical_reward \
        # -   x_rot_weight        * x_rot_reward \
        # -   y_rot_weight        * y_rot_reward \
        # -   z_rot_weight        * z_rot_reward

        forward = forward_weight    * forward_reward    
        lateral = lateral_weigth    * lateral_reward    * -1
        vertical = vertical_weigth  * vertical_reward   * -1
        x_rot = x_rot_weight        * x_rot_reward      * -1
        y_rot = y_rot_weight        * y_rot_reward      * -1
        z_rot = z_rot_weight        * z_rot_reward      * -1

        
        reward = 0 \
        + forward \
        + lateral \
        + vertical \
        + x_rot \
        + y_rot \
        + z_rot 

        reward_values = \
        {
            'forward'   : [forward],
            'lateral'   : [lateral],
            'vertical'  : [vertical],
            'x_rot'     : [x_rot],
            'y_rot'     : [y_rot],
            'z_rot'     : [z_rot],
            'reward'    : [reward]
        }


        return reward, reward_values


    def step_cb(msg):
        print("mydebug - step callback called!!!", flush=True)

        global first

        # global got_position
        # global got_orientiation
        # global got_joint_positions
        global got_state

        global actual_joint_positions
        global actual_pos
        global actual_ori

        global m2s
        global s2m

        # global ori_x
        # global ori_y
        # global ori_z
        # global pos_x
        # global pos_y
        # global pos_z

        global pos_shift_reg
        global ori_shift_reg
        global joint_positions_shift_reg

        global original_pos
        global original_ori

        #wait until all the state variables are known
        # while not got_position and not got_orientiation and not got_joint_positions:
        while not got_state:
            time.sleep(0.001)
            print("waiting for state, now it's ", got_state, flush=True)
        
        got_state = False
        #setting default values for next cycle
        # got_position = False
        # got_orientiation = False
        # got_joint_positions = False

        #send signal to publish new positions
        # actual_pos = [pos_x, pos_y, pos_z]
        pos_shift_reg = shift_elements(pos_shift_reg, actual_pos, 18, 3) #TODO generalize
        # actual_ori = [ori_x, ori_y, ori_z]
        ori_shift_reg = shift_elements(ori_shift_reg, actual_ori, 18, 3) #TODO generalize
        joint_positions_shift_reg = shift_elements(joint_positions_shift_reg, actual_joint_positions, 60, 20)

        state = [*pos_shift_reg, *ori_shift_reg, *joint_positions_shift_reg]

        
        if(is_fallen(actual_pos)):
            

            # while sim_state == 1:
            #     time.sleep(0.001)
                # print("waiting for restart")
            # print("restarted!!!")
            print("starting the reset process", flush=True)

            stop_publisher.publish(Bool(True))
            time.sleep(0.5)
            sync_publisher.publish(Bool(True))
            time.sleep(1)
            start_publisher.publish(Bool(True))  #start simulation
            time.sleep(1)

            # print("waiting before triggering next step", flush=True)
            # time.sleep(2)
            # print("waiting done")

            # got_state = False
            step_publisher.publish(Bool(True)) #trigger next step
            time.sleep(1)
            # print("finished the reset process\n\n", flush=True)
            # time.sleep(5)
            # print("check noww!!!!!!!")
            
            

            # got_state = False
            return




        # if first:
        #     if (actual_pos != [0,0,0]) and (actual_ori != [0,0,0]):
        #         m2s.put(state)
        #         first = 0
        #         original_pos = actual_pos
        #         original_ori = actual_ori
        #     else:
        #         print("Bad position and orientation initialization avoided!")
        #         return
        # else:
        #     m2s.put(state)
        #     reward, reward_values = reward_function(actual_pos, actual_ori)
        #     m2s.put(reward)
        #     q4.put(reward_values)
        #     done = is_it_done(actual_pos)
        #     m2s.put(done)

        # action = s2m.get(block=True)

        # # mapped_action = list(map(map_to_discrete_range, action[0].tolist()))

        # mapped_action = action[0].tolist()
        # mapped_action = [x / 0.05 for x in mapped_action]

        action_packet = Float32MultiArray()
        # action_packet.data = mapped_action

        action_packet.data = [1] * 20
    
        joint_publisher0.publish(action_packet)

        # print("waiting before triggering next step", flush=True)
        # time.sleep(2)
        # print("waiting done", flush=True)

        step_publisher.publish(z)
        # print("not fallen -> new step activated\n\n", flush=True)

        # time.sleep(1)
        
        return
    





    pos_shift_reg = [0] * 18
    ori_shift_reg = [0] * 18
    joint_positions_shift_reg = [0] * 60

    pause_flag = True
    sim_state = Float32MultiArray()

    #TODO nem mindegy hol van
    # position_cb_enable = True
    # orientation_cb_enable = True
    # joint_positions_cb_enable = True

    # dummy_joint_control("something")


    # mp.set_start_method('spawn')
    # manager = mp.Manager()
    # m2s = mp.Queue()
    # s2m = mp.Queue()

    # p1 = mp.Process(target=joint_control, args=(m2s, s2m), daemon=True)
    # # p1 = Process(target=joint_control, args=(q1,))
    # p1.start()

    # time.sleep(3) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1

    # q2 = Queue()
    # p2 = Process(target=process_image, args=(q2,))
    # p2.start()

    # q3 = Queue()
    # p3 = Process(target=graph_state, args=[])
    # p3.start()

    # q4 = Queue()
    # p4 = Process(target=graph_reward, args=[q4,])
    # p4.start()
    
    rospy.init_node('hugo_main')
    q_size = 10

    sync_publisher = rospy.Publisher("/enableSyncMode", Bool, queue_size=q_size)#, latch=True)
    start_publisher = rospy.Publisher("/startSimulation", Bool, queue_size=q_size)#, latch=True)
    stop_publisher = rospy.Publisher("/stopSimulation", Bool, queue_size=q_size)#, latch=True)
    step_publisher = rospy.Publisher("/triggerNextStep", Bool, queue_size=q_size)#, latch=True)
    puse_publisher = rospy.Publisher("/pauseSimulation", Bool, queue_size=q_size)
    joint_publisher0 = rospy.Publisher('/action', Float32MultiArray, queue_size=q_size)



    #rospy.Subscriber("/simulationState", Int32, simState_cb)
    rospy.Subscriber("/state", Float32MultiArray, state_cb, queue_size = q_size)
    rospy.Subscriber("/simulationStepDone", Bool, step_cb, queue_size = q_size)#, latch=True)
    

    
    time.sleep(0.1) #original value 5
    
    
    # print("mydebug - simulation started by main")
    z = Bool(True)

    delay = 0.3

    sync_publisher.publish(z)   #synchronize
    time.sleep(delay)
    start_publisher.publish(z)  #start simulation
    time.sleep(delay)
    step_publisher.publish(z)   #next step
    time.sleep(delay)
    step_publisher.publish(z) #needed because the simulator doesnt publish the states with only one step
    # time.sleep(delay)
    
    
    rate = rospy.Rate(10)

    time.sleep(0.5)
    print("initialization done", flush=True)


    # z.data = True

    signal.signal(signal.SIGINT, sigint_handler)
    signal.signal(signal.SIGQUIT, sigquit_handler)

    # signal.signal(signal.SIGSTOP, sigstop_handler) #TODO




    # time.sleep(2) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    print("main thread", flush=True)

    prev_state = 0
    counter = 0

    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        # print("main running")

        # #watchdog mechanism
        # if got_simstate:
        #     current_state = sim_state.data
        #     if (current_state == 0) and (prev_state == 0): #ha nem futott es nem is fut
        #         # print("first")
        #         counter += 1
        #         # print("counter in first = ", counter)
        #     elif (current_state == 1) and (prev_state == 0): #ha nem futott de most fut
        #         # print("second")
        #         counter = 0
        #         # print("counter in second = ", counter)

        #     if counter >= 40: #ha leallt
        #         # pass
        #         print("main restarts simulation!!!!!!")
        #         sync_publisher.publish(Bool(True))
        #         time.sleep(1)
        #         start_publisher.publish(Bool(True))  #start simulation
        #         time.sleep(1)
        #         step_publisher.publish(Bool(True)) #trigger next step
        #         time.sleep(1)
        #         counter = 0

        #     prev_state = current_state
        # else:
        #     print("No simstate yet!!")


        rate.sleep()

    