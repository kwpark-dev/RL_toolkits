#! /usr/bin/env python3

# Old collections usage
import collections.abc
collections.MutableMapping = collections.abc.MutableMapping
collections.MutableSequence = collections.abc.MutableSequence

################## Kinova kortex driver API #################
import sys
import os
import time
import threading
# See https://github.com/Kinovarobotics/Kinova-kortex2_Gen3_G3L
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2
################## Kinova kortex driver API #################

import numpy as np
import ffmpeg
import cv2
import torch
import matplotlib.pyplot as plt

from scipy import ndimage

from cognition.network import ResidualEncoder, ValueEncoder
from cognition.policy import StochasticActor
from cognition.value import CumRewardCritic
from cognition.buffer import RolloutBuffer
from cognition.agent import AgentPPO



# Maximum allowed waiting time during actions (in seconds)
TIMEOUT_DURATION = 20
# Image Config
WIDTH = 1280
HEIGHT = 720
COLOR = 3
# RTSP server
URL = "rtsp://192.168.1.10/color"
# Exp. Configuration
STEP = 16
EPISODE = 20
DISTURB = 0.25
# Agent Configuration
CONFIG = {}
CONFIG['buffer'] = {'name':RolloutBuffer,
                    'size':STEP}
CONFIG['actor'] = {'name':StochasticActor,
                   'model':ResidualEncoder,
                   'lr':1e-4}
CONFIG['critic'] = {'name':CumRewardCritic,
                    'model':ValueEncoder,
                    'lr':1e-3}
CONFIG['ppo'] = {'clip':0.3}
CONFIG['epoch'] = 12



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
    pose.x = 0.40 + 0.1*np.random.rand(1).item()
    pose.y = 0.000 + 0.05*np.random.rand(1).item()
    pose.z = 0.250 + 0.05*np.random.rand(1).item()
    pose.theta_x = 165 + 5*np.random.rand(1).item()
    pose.theta_y = 0 + 5*np.random.rand(1).item()
    pose.theta_z = 85 + 5*np.random.rand(1).item()

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



def sequential_movement(base, move):
    print("Starting sequential movement")
    
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
    #print(action)
    print("Waiting for movement to finish ...")
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        print("Cartesian movement completed")
    else:
        print("Timeout on action notification wait")
    return finished


def measure_pose(base):
    move = np.zeros(7)
    
    pose = base.GetMeasuredCartesianPose()
    
    gripper = Base_pb2.GripperRequest()
    gripper.mode = Base_pb2.GRIPPER_POSITION
    gripper_measure = base.GetMeasuredGripperMovement(gripper)
    finger = -1
    
    if gripper_measure.finger[0].value > 0.1:
        finger = 1

    move = (pose.x, pose.y, pose.z, pose.theta_x, pose.theta_y, pose.theta_z, finger)

    return move


def get_charge(base):    
    pose = base.GetMeasuredCartesianPose()
    
    gripper = Base_pb2.GripperRequest()
    gripper.mode = Base_pb2.GRIPPER_POSITION
    gripper_measure = base.GetMeasuredGripperMovement(gripper)
    finger = gripper_measure.finger[0].value

    reward = (pose.x + sigmoid(pose.y) + pose.z**3 +
              pose.theta_x/180 + sigmoid(pose.theta_y/180) + (pose.theta_z/180)**3 +
              finger - 10)

    return reward


def color_state(width, height):
    
    process = (
               ffmpeg.input(URL, rtsp_transport='tcp', t=1)
                     .output('pipe:', format='rawvideo', pix_fmt='bgr24', vframes=1)
                     .global_args('-loglevel', 'quiet')
                     .run_async(pipe_stdout=True)
              )

    in_bytes = process.stdout.read(WIDTH*HEIGHT*3)
    frame = np.frombuffer(in_bytes, np.uint8).reshape((HEIGHT, WIDTH, 3))

    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)



def sigmoid(x):
    return 1./(1.+np.exp(-x))



def main():
    
    # Import the utilities helper module
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import utilities

    # Parse arguments
    args = utilities.parseConnectionArguments()
    
    # data container
    cum_reward = []
    actor_loss_evol = []
    critic_loss_evol = []

    # Create connection to the device and get the router
    with utilities.DeviceConnection.createTcpConnection(args) as router:     
        # Create required services
        base = BaseClient(router)
                  
        # min, max of pose for robot safety
        mini = np.array([0.30, -0.20, 0.05, 60, -60, 60, -1])
        maxi = np.array([0.70, 0.2,0.5, 170, 60, 120, 1])
        interval = maxi - mini
        
        resize = 128
        state_dim = (resize, resize, COLOR)
        action_dim = 7

        agent = AgentPPO(state_dim, action_dim, CONFIG)
        trunc = STEP

        for ep in range(EPISODE):

            step = 0
            R = 0
            done = False
            success = True

            success &= initialize_position(base)
        
            state = color_state(resize, resize)
            #cv2.imwrite("images/ep_{}_state_{}.jpg".format(ep, step), state)
            # Data collection starts 
            while True:
                #move = np.random.rand(7)*interval + mini
                action, value, logp = agent.policy(torch.from_numpy(state).permute(2, 1, 0))
                # rescale
                move = sigmoid(action)*interval + mini
                #success &= sequential_movement(base, move)
                                
                dice = np.random.rand(1)
                if dice < DISTURB:
                    #print("old movement", move)
                    demo = input("press Enter after finish demo")
                    move = measure_pose(base)
                    print("new movement", move)

                success &= sequential_movement(base, move)
                charge = get_charge(base)
                incentive = input("Assign immediate incentive for the action: ")
                incentive = float(incentive)
                reward = charge + incentive
                print("Reward {} = charge {} + incentive {} ".format(incentive+charge, charge, incentive))
                
                next_state = color_state(resize, resize)

                agent.buffer.push(state, action, reward, value, logp)
                state = next_state

                R += float(reward)
                step += 1
            
                #frame = color_state(resize, resize)
                #cv2.imwrite("images/ep_{}_state_{}.jpg".format(ep, step), frame)

                print("step {} is done".format(step))
                if step >= trunc:
                    print("Episode is done")
                    success &= example_move_to_home_position(base)
                    break

                if int(incentive) == 50:
                    done = True
                    success &= example_move_to_home_position(base)
                    break

            _, last_value, _ = agent.policy(torch.from_numpy(state).permute(2, 1, 0))

            if done:
                last_value = None
            # Training starts
            actor_loss, critic_loss = agent.learn(last_value=last_value)
            # Note that actor loss = policy loss + context*0.5
            cum_reward.append(R)
            actor_loss_evol.append(actor_loss)
            critic_loss_evol.append(critic_loss)

            print("Cumulative reward is {} at Episode {}".format(R, ep))
            print("Actor loss: {}, Critic loss: {}".format(actor_loss, critic_loss))

            # Test starts
            print("\n NOW TEST MODE \n")
            
            step = 0
            success &= initialize_position(base)
            state = color_state(resize, resize)
            snapshot = state.copy()
            #cv2.imwrite("data/state_after_ep_{}_step_{}.jpg".format(ep, step), state)
            
            while True:
                snapshot = state.copy()
                action, val, f1, w1, f2, w2 = agent.policy(torch.from_numpy(state).permute(2, 1, 0), True)

                fig, ax = plt.subplots(1, 2, figsize=(10, 4))
                cam1 = f1@w1
                cam2 = f2@w2
                cam1 = ndimage.zoom(cam1, 4, order=2)
                cam2 = ndimage.zoom(cam2, 4, order=2)
                
                ax[0].imshow(snapshot, alpha=0.6)
                ax[0].imshow(cam1, cmap='jet', alpha=0.4)
                ax[1].imshow(snapshot, alpha=0.6)
                ax[1].imshow(cam2, cmap='jet', alpha=0.4)
                plt.savefig("images/test_ep_{}_step_{}.jpg".format(ep, step))
                #plt.show()

                move = sigmoid(action)*interval + mini
                print(move)
                print("This move has {} value".format(value))
                success &= sequential_movement(base, move)
                next_state = color_state(resize, resize)

                if step >= trunc:
                    print("TEST episode ends")
                    success &= example_move_to_home_position(base)
                    break

                step += 1
                state = next_state

        np.save('data/cum_reward.npy', np.array(cum_reward))
        np.save('data/actor_loss_evol.npy', np.array(actor_loss_evol))
        np.save('data/critic_loss_evol.npy', np.array(critic_loss_evol))

        agent.save_model('data/checkpoints/null')

        return 0 if success else 1


if __name__ == "__main__":
    exit(main())
