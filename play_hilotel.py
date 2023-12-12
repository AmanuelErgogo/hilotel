import rospy
import argparse
import json
import os
import time


# add argparse arguments
parser = argparse.ArgumentParser("Welcome to hilotel: vr based human in the loop learning and execution platform!")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Pytorch model checkpoint to load.")
args_cli = parser.parse_args()

"""Rest everything follows."""

import gymnasium as gym
import torch
import numpy as np

import robomimic  # noqa: F401
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

from xarm7_env import XArm7Env

expert_actions = [0.0, 0.0, 0.0, 0.0]
is_interrupt = False

action_type = "absolute" 
stop = 0
is_update = False
done_trigger = 0
action = [0.0, 0.0, 0.0, 0.0]
previous_command = np.array([0.0, 0.0, 0.0])

hilotel_execution_time = {}

def joy_cmd_clb(msg):
    global expert_actions, is_interrupt
    """
    R_trigger: open/close gripper 
    R_B: Stop
    R_A: Done
    """
    expert_actions[3] = msg.axes[2]
    # self.done_trigger = 1 if msg.buttons[2] == 1 else 0
    if msg.buttons[3] == 1:
        stop = 1
    if msg.buttons[0] == 1:
        is_interrupt = True

def oculus_delta_pose_cb(msg):
    """Obtains expert action from oculus quest 2 interface."""
    global action_type, previous_command, is_update
    r_controller_pos = [msg.pos_z, -msg.pos_x, msg.pos_y]
    if action_type == "relative":
        """action = delta_pos between -1 and 1
        - calculate delta 
        - normalize between -1 and 1  to make it similar with agents action
        """
        if not is_update: # first command
            previous_command = np.array(r_controller_pos).copy()
        delta_command = np.array(r_controller_pos) - previous_command
        normalized_delta_command = np.clip(delta_command, -1.0, 1.0)
        previous_command = np.array(r_controller_pos).copy()
        expert_actions[0] = normalized_delta_command[0]
        expert_actions[1] = normalized_delta_command[1]
        expert_actions[2] = normalized_delta_command[2]
    else: 
        expert_actions[0] = msg.pos_z
        expert_actions[1] = -msg.pos_x
        expert_actions[2] = msg.pos_y
    is_update = True

def main():
    global expert_actions, is_interrupt
    """Run a trained policy from robomimic with ufactory xarm7 environment."""
    rospy.init_node('hilotel_eval_node', anonymous=True)

    rate = rospy.Rate(50)

    # register env
    gym.envs.register(
        id='XArm7Env-v1',
        entry_point='xarm7_env_robomimic:XArm7Env',
    )

    rng = np.random.default_rng(0)
    env = gym.make('XArm7Env-v1')

    # acquire device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    # restore policy
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=args_cli.checkpoint, device=device, verbose=True)

    # reset environment
    while True:
        obs, info = env.reset()
        if info['is_init']:
            break
        print("obs:", obs)
        print("info:", info)
    while not rospy.is_shutdown():
        if is_new_demo:
            user_input = input("Enter to continue next eval: Any key")
            is_new_demo = False
            demo_count = demo_count + 1
            t_start = time.monotonic()
        # compute actions
        if is_interrupt:
            actions = policy(obs)
        else:
            actions = expert_actions
        print("actions:", actions)
        print("obs: ", obs)
        # apply actions
        obs, reward, terminated, truncated, info= env.step(actions)

        dones = np.array([info["is_success"]])
        
        if dones[0] == True:
            hilotel_execution_time[f"demo_{demo_count}_min"] = (time.monotonic() - t_start) / 60
            is_new_demo = True
            
            log_dir_exec_time = os.path.join("./logs/robomimic", args_cli.task, 'agent_execution_time.json')
            with open(log_dir_exec_time, 'w') as json_file:
                json.dump(hilotel_execution_time, json_file)

            # reset env to start second evaluation
            max_executions = 300
            counter = 0
            while counter < max_executions:
                print ("resseting>>>")
                env.step(action=actions)
                counter += 1
                if counter >= max_executions:
                    info["is_success"] = False
                    break

        rate.sleep()

    env.close()


if __name__ == "__main__":
    main()