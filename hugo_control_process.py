#!/usr/bin/env python3

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
import matplotlib
# matplotlib.use("Qt5agg")
# matplotlib.use('Agg')


# Instantiate CvBridge
bridge = CvBridge()

position_cb_enable = True
orientation_cb_enable = True

got_position = True
got_orientiation = True



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



def simTime_cb(msg):
    # print("Simulation time: ",msg)
    return



def simState_cb(msg):
    # print("Simulation state: ",msg)
    return

#------------------------------------------------------------------------------------------------------------

def joint_control(q):
    rospy.init_node('hugo_control')

    q_size = 10

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

    i = 0
    while not rospy.is_shutdown():

        # print(q.empty())
        # print("hello hello")
        # if q.empty():
        #     print("Nothing in queue of joint_control")
        # nothing = q.get()

        # print(q.get())
        q.get()

        #NAIVE CONTROL
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        joint_pos = [math.sin(i)] * 26
        i += 0.1
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


        #NEURAL NETWORK CONTROL
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
        joint_publisher1.publish(joint_pos[0])
        joint_publisher2.publish(joint_pos[1])
        joint_publisher3.publish(joint_pos[2])
        joint_publisher4.publish(joint_pos[3])
        joint_publisher5.publish(joint_pos[4])
        joint_publisher6.publish(joint_pos[5])
        joint_publisher7.publish(joint_pos[6])
        joint_publisher8.publish(joint_pos[7])
        joint_publisher9.publish(joint_pos[8])
        joint_publisher10.publish(joint_pos[9])
        joint_publisher11.publish(joint_pos[10])
        joint_publisher12.publish(joint_pos[11])
        joint_publisher13.publish(joint_pos[12])
        joint_publisher14.publish(joint_pos[13])
        joint_publisher15.publish(joint_pos[14])
        joint_publisher16.publish(joint_pos[15])
        joint_publisher17.publish(joint_pos[16])
        joint_publisher18.publish(joint_pos[17])
        joint_publisher19.publish(joint_pos[18])
        joint_publisher20.publish(joint_pos[19])
        joint_publisher21.publish(joint_pos[20])
        joint_publisher22.publish(joint_pos[21])
        joint_publisher23.publish(joint_pos[22])
        joint_publisher24.publish(joint_pos[23])
        joint_publisher25.publish(joint_pos[24])
        joint_publisher26.publish(joint_pos[25])

        q.put("hello from joint_control")
        # print("joint control thread joint control thread joint control thread ")
        rate.sleep()
        # rospy.sleep(0.1)

#------------------------------------------------------------------------------------------------------------

def process_image(q):

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

time_list = [0]
pos_x_list = [0]
pos_y_list = [0]
pos_z_list = [0]
ori_x_list = [0]
ori_y_list = [0]
ori_z_list = [0]


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
        ax1.set_xlim(0, time_list[-1])

        ax2.set_ylim(padded_pos_min, padded_pos_max)
        ax2.set_xlim(0, time_list[-1])

        ax3.set_ylim(padded_pos_min, padded_pos_max)
        ax3.set_xlim(0, time_list[-1])

        ax4.set_ylim(padded_ori_min, padded_ori_max)
        ax4.set_xlim(0, time_list[-1])

        ax5.set_ylim(padded_ori_min, padded_ori_max)
        ax5.set_xlim(0, time_list[-1])

        ax6.set_ylim(padded_ori_min, padded_ori_max)
        ax6.set_xlim(0, time_list[-1])

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

    


if __name__ == '__main__':

    q1 = Queue()
    p1 = Process(target=joint_control, args=(q1,))
    p1.start()

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
    

    def step_cb(msg):
        global got_position
        global got_orientiation
        global q1

        # print("step done")

        while not got_position and not got_orientiation:
            time.sleep(0.001)

        # print(got_position, got_orientiation)
        
        #setting default values for next cycle
        got_position = False
        got_orientiation = False

        #send signal to publish new positions
        q1.put("hello from main")

        #wait for the response
        response = q1.get()
        # print(response)

        # time.sleep(0.5)

        #we can make a step with the simulator
        step_publisher.publish(z)
        # time.sleep(5)
        
        return
    

    #Ctrl-C handling
    def signal_handler(*args):
        print("\nexiting!!!")
        stop_publisher.publish(z)
        p1.kill()
        p2.kill()
        p1.join()
        p2.join()
        print("Processes should be joined by now")
        exit(0)

    def visu_cb(msg):
        print(msg.data[0])
        print("callback working")

    rospy.Subscriber("/simulationStepDone", Bool, step_cb, queue_size = q_size)#, latch=True)
    rospy.Subscriber("/simulationTime", Float32, simTime_cb, queue_size = q_size)
    rospy.Subscriber('/robPosition', Point32, position_cb, queue_size = q_size)
    rospy.Subscriber('/robOrientation', Point32, orientation_cb, queue_size = q_size)
    rospy.Subscriber("/simulationState", Int32, simState_cb)


    position_cb_enable = True
    orientation_cb_enable = True

    time.sleep(0.5)
    print("initialization done")

    z = Bool(True)
    # z.data = True

    signal.signal(signal.SIGINT, signal_handler)
    sync_publisher.publish(z)   #synchronize
    time.sleep(0.1)
    start_publisher.publish(z)  #start simulation
    time.sleep(0.1)
    step_publisher.publish(z)   #next step
    time.sleep(0.1) #original value 5


    time.sleep(5)


    print("main thread")

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        # print("main thread main thread main thread main thread")
        # q1.put("hello")
        rate.sleep()

    stop_publisher.publish(z)