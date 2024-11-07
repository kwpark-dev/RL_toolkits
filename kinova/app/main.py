#! /usr/bin/env python3

###
# KINOVA (R) KORTEX (TM)
#
# Copyright (c) 2018 Kinova inc. All rights reserved.
#
# This software may be modified and distributed
# under the terms of the BSD 3-Clause license.
#
# Refer to the LICENSE file for details.
#
###

import collections.abc
collections.MutableMapping = collections.abc.MutableMapping
collections.MutableSequence = collections.abc.MutableSequence

import sys
import os
import time
import threading

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2

from enum import Enum

import numpy as np
import ffmpeg
import cv2

# Maximum allowed waiting time during actions (in seconds)
TIMEOUT_DURATION = 20
# Image Config
WIDTH = 1280
HEIGHT = 720
COLOR = 3
# RTSP server
URL = "rtsp://192.168.1.10/color"


class PoseSafety(Enum):
    XMIN = 30
    XMX = 70
    YMIN = -20
    YMAX = 20
    ZMIN = 4
    ZMAX = 54

    THETAXMIN = 60
    THETAXMAX = 180
    THETAYMIN = -60
    THETAYMAX = 60
    THETAZMIN = 60
    THETAZMAX = 120


# Create closure to set an event after an END or an ABORT
def check_for_end_or_abort(e):
    """Return a closure checking for END or ABORT notifications

    Arguments:
    e -- event to signal when the action is completed
        (will be set when an END or ABORT occurs)
    """
    def check(notification, e = e):
        print("EVENT : " + \
              Base_pb2.ActionEvent.Name(notification.action_event))
        if notification.action_event == Base_pb2.ACTION_END \
        or notification.action_event == Base_pb2.ACTION_ABORT:
            e.set()
    return check
 
def example_move_to_home_position(base):
    # Make sure the arm is in Single Level Servoing mode
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)
    
    # Move arm to ready position
    print("Moving the arm to a safe position")
    action_type = Base_pb2.RequestedActionType()
    action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
    action_list = base.ReadAllActions(action_type)
    action_handle = None
    for action in action_list.action_list:
        if action.name == "Home":
            action_handle = action.handle

    if action_handle == None:
        print("Can't reach safe position. Exiting")
        return False

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    base.ExecuteActionFromReference(action_handle)
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        print("Safe position reached")
    else:
        print("Timeout on action notification wait")
    return finished


def initialize_position(base):
    # Make sure the arm is in Single Level Servoing mode
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)

    # cartesian action movement, end_effector control
    print("Initialize Position: Gripper")
    gripper_command = Base_pb2.GripperCommand()
    finger = gripper_command.gripper.finger.add()

    gripper_command.mode = Base_pb2.GRIPPER_POSITION
    finger.finger_identifier = 1
    finger.value = 0.
    
    print(gripper_command) 
    base.SendGripperCommand(gripper_command)
    time.sleep(1)
    
    print("Initialize Position: Joints")
    action = Base_pb2.Action()
    action.name = "Initial Pose"
    action.application_data = ""

    pose = action.reach_pose.target_pose
    pose.x = 0.420
    pose.y = 0.000
    pose.z = 0.455
    pose.theta_x = 178.8
    pose.theta_y = 0
    pose.theta_z = 90

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    print("Executing action")
    base.ExecuteAction(action)
    
    print(action)
    
    print("Waiting for movement to finish ...")
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        print("Initializtion Completed")
    else:
        print("Timeout on action notification wait")
    return finished


def example_angular_action_movement(base):
    
    print("Starting angular action movement ...")
    action = Base_pb2.Action()
    action.name = "Example angular action movement"
    action.application_data = ""

    actuator_count = base.GetActuatorCount()

    # Place arm straight up
    for joint_id in range(actuator_count.count):
        joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
        joint_angle.joint_identifier = joint_id
        joint_angle.value = 10

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )
    
    print("Executing action")
    base.ExecuteAction(action)
    print(action)
    print("Waiting for movement to finish ...")
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        print("Angular movement completed")
    else:
        print("Timeout on action notification wait")
    return finished


def sequential_movement(base, move):
    print("Starting sequentail movement")
    
    gripper_command = Base_pb2.GripperCommand()
    finger = gripper_command.gripper.finger.add()

    gripper_command.mode = Base_pb2.GRIPPER_POSITION
    context = 1/(1+np.exp(-move[-1]))
    print("probability is {} \n".format(context)) 
    if context > 0.5:
        finger.value = 0.4
   
    else:
        finger.value = 0
    
    print(gripper_command)

    base.SendGripperCommand(gripper_command)
    time.sleep(1)

    action = Base_pb2.Action()
    action.name = "grasp the target"
    action.application_data = ""

    pose = action.reach_pose.target_pose
    pose.x, pose.y, pose.z, pose.theta_x, pose.theta_y, pose.theta_z = move[:6]

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    print("Executing action")
    base.ExecuteAction(action)
    print(action)
    print("Waiting for movement to finish ...")
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        print("Cartesian movement completed")
    else:
        print("Timeout on action notification wait")
    return finished


def example_cartesian_action_movement(base, base_cyclic):
    
    print("Starting Cartesian action movement ...")
    action = Base_pb2.Action()
    action.name = "Example Cartesian action movement"
    action.application_data = ""

    feedback = base_cyclic.RefreshFeedback()

    cartesian_pose = action.reach_pose.target_pose
    cartesian_pose.x = 0.388
    cartesian_pose.y = 0.001
    cartesian_pose.z = 0.278
    cartesian_pose.theta_x = 178.3
    cartesian_pose.theta_y = 0
    cartesian_pose.theta_z = 90
    #cartesian_pose.x = feedback.base.tool_pose_x          # (meters)
    #cartesian_pose.y = feedback.base.tool_pose_y - 0.1    # (meters)
    #cartesian_pose.z = feedback.base.tool_pose_z - 0.2    # (meters)
    #cartesian_pose.theta_x = feedback.base.tool_pose_theta_x # (degrees)
    #cartesian_pose.theta_y = feedback.base.tool_pose_theta_y # (degrees)
    #cartesian_pose.theta_z = feedback.base.tool_pose_theta_z # (degrees)

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    print("Executing action")
    base.ExecuteAction(action)
    print(action)
    print("Waiting for movement to finish ...")
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        print("Cartesian movement completed")
    else:
        print("Timeout on action notification wait")
    return finished


def gripper_movement(base):

    gripper_command = Base_pb2.GripperCommand()
    finger = gripper_command.gripper.finger.add()

    gripper_command.mode = Base_pb2.GRIPPER_POSITION
    finger.finger_identifier = 1
    finger.value = 0.
    
    print("Gripper moves to {}".format(finger.value))
    base.SendGripperCommand(gripper_command)
        
    time.sleep(1)
    
    return 0


def color_state(width, height):
    
    process = (
               ffmpeg.input(URL, rtsp_transport='tcp', t=1)
                     .output('pipe:', format='rawvideo', pix_fmt='bgr24', vframes=1)
                     .run_async(pipe_stdout=True)
              )

    in_bytes = process.stdout.read(WIDTH*HEIGHT*3)
    frame = np.frombuffer(in_bytes, np.uint8).reshape((HEIGHT, WIDTH, 3))

    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)


def main():
    
    # Import the utilities helper module
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import utilities

    # Parse arguments
    args = utilities.parseConnectionArguments()
      
    # Create connection to the device and get the router
    with utilities.DeviceConnection.createTcpConnection(args) as router:     
        # Create required services
        base = BaseClient(router)
        
        step = 0
        R = 0
        success = True
        
        # min, max of pose for robot safety
        mini = np.array([0.30, -0.20, 0.05, 60, -60, 60, 0])
        maxi = np.array([0.70, 0.2,0.54, 180, 60, 120, 1])
        interval = maxi - mini

        success &= initialize_position(base)
        
        frame = color_state(128, 128)
        cv2.imwrite("images/state%d.jpg" % step, frame)

        while True:
            move = np.random.rand(7)*interval + mini
            success &= sequential_movement(base, move)
            
            reward = input("Assign immediate reward for the action: ")
            print("Reward {} was assigned".format(reward))
            
            R += float(reward)
            step += 1
            
            frame = color_state(128, 128)
            cv2.imwrite("images/state%d.jpg" % step, frame)

            print("step {} is done".format(step))
            if step > 2:
                print("Episode is done")
                success &= example_move_to_home_position(base)

                break

        print("Cumulative reward is {}".format(R))

        return 0 if success else 1


if __name__ == "__main__":
    exit(main())
