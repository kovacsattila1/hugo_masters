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
# import torch.multiprocessing.queue as queue
# from multiprocessing import Manager

logger = logging.getLogger('train')
logger.setLevel(logging.DEBUG)

# Instantiate CvBridge
bridge = CvBridge()

position_cb_enable = True
orientation_cb_enable = True

got_position = True
got_orientiation = True

time_list = [0]
pos_x_list = [0]
pos_y_list = [0]
pos_z_list = [0]
ori_x_list = [0]
ori_y_list = [0]
ori_z_list = [0]

pos_x = 0
pos_y = 0
pos_z = 0
ori_x = 0
ori_y = 0
ori_z = 0




#------------------------------------------------------------------------------------------------------------

def joint_control(m2s, s2m):
    print("torch process started!!!", flush=True)
    # rospy.init_node('hugo_control')
    # q_size = 10
    # rospy.Subscriber("/simulationStepDone", Bool, step_cb, queue_size = q_size)#, latch=True)


    writer = SummaryWriter('runs/run_1')
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
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
    observation_space_dim = 6
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
            action = agent.calc_action(state, ou_noise)
            print("joint control action shape", action.shape)

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
                agent.save_checkpoint(timestep, memory)
                time_last_checkpoint = time.time()
                logger.info('Saved model at {}'.format(time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))

        epoch += 1

    agent.save_checkpoint(timestep, memory)
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


def graph_values():


    rospy.init_node('graph_values')
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
        lines[0].set_xdata(time_list)
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


if __name__ == '__main__':


    first = 1

    def map_to_discrete_range(value):
        input_min, input_max = -1, 1
        output_min, output_max = 0, 180

        # Clip the value to the input range
        value = np.clip(value, input_min, input_max)
        
        # Normalize the value to a 0-1 range
        normalized_value = (value - input_min) / (input_max - input_min)
        
        # Scale to the output range and round to the nearest integer
        discrete_value = np.round(normalized_value * (output_max - output_min) + output_min).astype(float)
        
        return discrete_value


    #Ctrl-C handling
    def sigint_handler(*args):
        print("\nexiting!!!")
        stop_publisher.publish(z)
        p1.kill()
        p2.kill()
        p1.join()
        p2.join()
        print("Processes should be joined by now")
        exit(0)



    def sigquit_handler(*args):
        print("You pressed Ctrl + \\")


    # def sigstop_handler(*args):
    #     print("You pressed Ctrl + Z")


    def position_cb(data):
        if position_cb_enable:
            global pos_x
            global pos_y
            global pos_z
            global got_position

            pos_x = data.x
            pos_y = data.y
            pos_z = data.z

            got_position = True
            # print("Position   : x:", data.x, "   y:", data.y, "   :", data.z)
    


    def orientation_cb(data):
        if orientation_cb_enable:
            global ori_x
            global ori_y
            global ori_z
            global got_orientiation 

            ori_x = math.degrees(data.x)
            ori_y = math.degrees(data.y)
            ori_z = math.degrees(data.z)

            ori_x = data.x
            ori_y = data.y
            ori_z = data.z
            
            got_orientiation = True
            # print("Orientation   : x:", data.x, "   y:", data.y, "   :", data.z)



    def simTime_cb(msg):
        # print("Simulation time: ",msg)
        return



    def simState_cb(msg):
        # print("Simulation state: ",msg)
        return
    

    def reward_function(state):
        return 1
    

    def is_it_done(state):
        return 0
    

    def step_cb(msg):
        global first
        global got_position
        global got_orientiation
        global m2s
        global s2m
        global ori_x
        global ori_y
        global ori_z
        global pos_x
        global pos_y
        global pos_z


        while not got_position and not got_orientiation:
            time.sleep(0.001)

        # print(got_position, got_orientiation)
        
        #setting default values for next cycle
        got_position = False
        got_orientiation = False

        #send signal to publish new positions
        state = [pos_x, pos_y, pos_z, ori_x, ori_y, ori_z]

        if first:
            m2s.put(state)
            first = 0
        else:
            m2s.put(state)
            reward = reward_function(state)
            m2s.put(reward)
            done = is_it_done(state)
            m2s.put(done)

        action = s2m.get(block=True)

        mapped_action = list(map(map_to_discrete_range, action[0].tolist()))

        joint_publisher1.publish(mapped_action[0])
        joint_publisher2.publish(mapped_action[1])
        joint_publisher3.publish(mapped_action[2])
        joint_publisher4.publish(mapped_action[3])
        joint_publisher5.publish(mapped_action[4])
        joint_publisher6.publish(mapped_action[5])
        joint_publisher7.publish(mapped_action[6])
        joint_publisher8.publish(mapped_action[7])
        joint_publisher9.publish(mapped_action[8])
        joint_publisher10.publish(mapped_action[9])
        joint_publisher11.publish(mapped_action[10])
        joint_publisher12.publish(mapped_action[11])
        joint_publisher13.publish(mapped_action[12])
        joint_publisher14.publish(mapped_action[13])
        joint_publisher15.publish(mapped_action[14])
        joint_publisher16.publish(mapped_action[15])
        joint_publisher17.publish(mapped_action[16])
        joint_publisher18.publish(mapped_action[17])
        joint_publisher19.publish(mapped_action[18])
        joint_publisher20.publish(mapped_action[19])
        joint_publisher21.publish(mapped_action[20])
        joint_publisher22.publish(mapped_action[21])
        joint_publisher23.publish(mapped_action[22])
        joint_publisher24.publish(mapped_action[23])
        joint_publisher25.publish(mapped_action[24])
        joint_publisher26.publish(mapped_action[25])


        step_publisher.publish(z)
        
        return
    








    # dummy_joint_control("something")


    mp.set_start_method('spawn')
    # manager = mp.Manager()
    m2s = mp.Queue()
    s2m = mp.Queue()

    p1 = mp.Process(target=joint_control, args=(m2s, s2m), daemon=True)
    # p1 = Process(target=joint_control, args=(q1,))
    p1.start()

    # time.sleep(3) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1

    q2 = Queue()
    p2 = Process(target=process_image, args=(q2,))
    p2.start()

    q3 = Queue()
    p3 = Process(target=graph_values, args=[])
    p3.start()
    
    rospy.init_node('hugo_main')
    q_size = 10

    sync_publisher = rospy.Publisher("/enableSyncMode", Bool, queue_size=q_size)#, latch=True)
    start_publisher = rospy.Publisher("/startSimulation", Bool, queue_size=q_size)#, latch=True)
    stop_publisher = rospy.Publisher("/stopSimulation", Bool, queue_size=q_size)#, latch=True)
    step_publisher = rospy.Publisher("/triggerNextStep", Bool, queue_size=q_size)#, latch=True)
    
    rospy.Subscriber("/simulationStepDone", Bool, step_cb, queue_size = q_size)#, latch=True)
    rospy.Subscriber("/simulationTime", Float32, simTime_cb, queue_size = q_size)
    rospy.Subscriber('/robPosition', Point32, position_cb, queue_size = q_size)
    rospy.Subscriber('/robOrientation', Point32, orientation_cb, queue_size = q_size)
    rospy.Subscriber("/simulationState", Int32, simState_cb)


    joint_publisher1 = rospy.Publisher('/head_x_joint', Float32, queue_size=q_size)
    joint_publisher2 = rospy.Publisher('/head_y_joint', Float32, queue_size=q_size)
    joint_publisher3 = rospy.Publisher('/head_z_joint', Float32, queue_size=q_size)
    joint_publisher4 = rospy.Publisher('/right_shoulder_rotate_joint', Float32, queue_size=q_size)
    joint_publisher5 = rospy.Publisher('/right_shoulder_sideways_joint', Float32, queue_size=q_size)
    joint_publisher6 = rospy.Publisher('/right_elbow_joint', Float32, queue_size=q_size)
    joint_publisher7 = rospy.Publisher('/right_wrist_joint', Float32, queue_size=q_size)
    joint_publisher8 = rospy.Publisher('/left_shoulder_rotate_joint', Float32, queue_size=q_size)
    joint_publisher9 = rospy.Publisher('/left_shoulder_sideways_joint', Float32, queue_size=q_size)
    joint_publisher10 = rospy.Publisher('/left_elbow_joint', Float32, queue_size=q_size)
    joint_publisher11 = rospy.Publisher('/left_wrist_joint', Float32, queue_size=q_size)
    joint_publisher12 = rospy.Publisher('/waist_x_joint', Float32, queue_size=q_size)
    joint_publisher13 = rospy.Publisher('/waist_y_joint', Float32, queue_size=q_size)
    joint_publisher14 = rospy.Publisher('/waist_z_joint', Float32, queue_size=q_size)
    joint_publisher15 = rospy.Publisher('/right_hip_sideways_joint', Float32, queue_size=q_size)
    joint_publisher16 = rospy.Publisher('/right_hip_forward_joint', Float32, queue_size=q_size)
    joint_publisher17 = rospy.Publisher('/right_thight_joint', Float32, queue_size=q_size)
    joint_publisher18 = rospy.Publisher('/right_knee_joint', Float32, queue_size=q_size)
    joint_publisher19 = rospy.Publisher('/right_upper_ankle_joint', Float32, queue_size=q_size)
    joint_publisher20 = rospy.Publisher('/right_lower_ankle_joint', Float32, queue_size=q_size)
    joint_publisher21 = rospy.Publisher('/left_hip_sideways_joint', Float32, queue_size=q_size)
    joint_publisher22 = rospy.Publisher('/left_hip_forward_joint', Float32, queue_size=q_size)
    joint_publisher23 = rospy.Publisher('/left_thight_joint', Float32, queue_size=q_size)
    joint_publisher24 = rospy.Publisher('/left_knee_joint', Float32, queue_size=q_size)
    joint_publisher25 = rospy.Publisher('/left_upper_ankle_joint', Float32, queue_size=q_size)
    joint_publisher26 = rospy.Publisher('/left_lower_ankle_joint', Float32, queue_size=q_size)
    rate = rospy.Rate(10)


    position_cb_enable = True
    orientation_cb_enable = True

    time.sleep(0.5)
    print("initialization done")

    z = Bool(True)
    # z.data = True

    signal.signal(signal.SIGINT, sigint_handler)
    signal.signal(signal.SIGQUIT, sigquit_handler)
    # signal.signal(signal.SIGSTOP, sigstop_handler) #TODO

    sync_publisher.publish(z)   #synchronize
    time.sleep(0.1)
    start_publisher.publish(z)  #start simulation
    time.sleep(0.1)
    step_publisher.publish(z)   #next step
    time.sleep(0.1) #original value 5


    # time.sleep(2) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    print("main thread")

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        # print("main thread main thread main thread main thread")
        rate.sleep()

    stop_publisher.publish(z)