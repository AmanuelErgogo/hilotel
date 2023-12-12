import rospy
import argparse
import os
import time
import json

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

is_new_demo = True
demo_count = 0

execution_time = {}

def main():
    """Run a trained policy from robomimic with ufactory xarm7 environment."""
    rospy.init_node('hilotel_offline_policy_node', anonymous=True)

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
            user_input = input("Enter to continue demo: Any key")
            is_new_demo = False
            demo_count = demo_count + 1
            t_start = time.monotonic()
        # compute actions
        actions = policy(obs)
      
        # apply actions
        obs, reward, terminated, truncated, info= env.step(actions)
        dones = np.array([info["is_success"]])
        
        if dones[0] == True:
            execution_time[f"demo_{demo_count}_min"] = (time.monotonic() - t_start) / 60
            is_new_demo = True
            
            log_dir_exec_time = os.path.join("./logs/robomimic", args_cli.task, 'agent_execution_time.json')
            with open(log_dir_exec_time, 'w') as json_file:
                json.dump(execution_time, json_file)

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