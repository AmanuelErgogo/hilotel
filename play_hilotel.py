import rospy
import argparse

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
        # compute actions
        actions = policy(obs)
        print("actions:", actions)
        print("obs: ", obs)
        # apply actions
        obs, reward, terminated, truncated, info= env.step(actions)

        rate.sleep()

    env.close()


if __name__ == "__main__":
    main()