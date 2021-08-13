#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image # ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError # OpenCV2 for saving an image
import cv2
import math
from std_msgs.msg import Float32
from std_msgs.msg import Bool
from std_msgs.msg import Int32
from geometry_msgs.msg import Point32
import time
from multiprocessing import Process, Queue
import signal
import _thread

# Instantiate CvBridge
bridge = CvBridge()

def position_cb(data):
    # global pos_x
    # global pos_y
    # global pos_z
    # pos_x = data.x
    # pos_y = data.y
    # pos_z = data.z
    print("Position   : x:", data.x, "   y:", data.y, "   :", data.z)
    

def orientation_cb(data):
    # global ori_x
    # global ori_y
    # global ori_z
    # ori_x = math.degrees(data.x)end_point_charge
    # ori_y = math.degrees(data.y)
    # ori_z = math.degrees(data.z)
    # ori_x = data.x
    # ori_y = data.y
    # ori_z = data.z
    print("Orientation   : x:", data.x, "   y:", data.y, "   :", data.z)


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


def joint_control(q):
    # rospy.init_node('hugo_control')

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

    # rate = rospy.Rate(10)

    i = 0
    # while not rospy.is_shutdown():
    while True:
        joint_pos = [math.sin(i)] * 26
        i += 0.1
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
        time.sleep(0.1)
        # rate.sleep()


def process_image(q):
    # rospy.init_node('hugo_vision')
    q_size = 10
    rospy.Subscriber("/image", Image, image_cb, queue_size=q_size, buff_size=2**24)

    # rate = rospy.Rate(10)
    # while not rospy.is_shutdown():
    while True:
        time.sleep(0.1)
        # rate.sleep()










if __name__ == '__main__':


    def signal_handler(*args):
        print("\nexiting!!!")
        stop_publisher.publish(z)
        # p1.terminate()
        # p2.terminate()
        # p1.join()
        # p2.join()
        exit(0)


    def step_cb(msg):
        # print("step done")
        step_publisher.publish(z)
        return

    z = Bool(True)
    # z.data = True


    # q1 = Queue()
    # p1 = Process(target=joint_control, args=(q1,))
    # p1.start()

    # q2 = Queue()
    # p2 = Process(target=process_image, args=(q2,))
    # p2.start()

    # try:
    t1 = _thread.start_new_thread(joint_control,(1,))
    t1 = _thread.start_new_thread(process_image,(1,))
    # except:
    # print ("Error: unable to start thread")


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

    time.sleep(0.5)
    print("inicialization done")



    signal.signal(signal.SIGINT, signal_handler)
    sync_publisher.publish(z)   #synchronize
    time.sleep(0.1)
    start_publisher.publish(z)  #start simulation
    time.sleep(0.1)
    step_publisher.publish(z)   #next step


    print("main thread")

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        rate.sleep()

    stop_publisher.publish(z)


    



