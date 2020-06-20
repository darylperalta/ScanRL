""" Load and display pre-trained model in OpenAI Gym Environment
"""

import os
import sys
import gym
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K

from DDQN.ddqn import DDQN
from DDPG.ddpg import DDPG

from keras.backend.tensorflow_backend import set_session
from keras.utils import to_categorical

from utils.atari_environment import AtariEnvironment
# from utils.continuous_environments import Environment
from utils.unreal_environments import Environment
from utils.networks import get_session
import gym_unrealcv

# gym.logger.set_level(40)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def parse_args(args):
    """ Parse arguments from command line input
    """
    parser = argparse.ArgumentParser(description='Training parameters')
    #
    parser.add_argument('--type', type=str, default='DDQN',help="Algorithm to train from {A2C, A3C, DDQN, DDPG}")
    parser.add_argument('--is_atari', dest='is_atari', action='store_true', help="Atari Environment")
    parser.add_argument('--with_PER', dest='with_per', action='store_true', help="Use Prioritized Experience Replay (DDQN + PER)")
    parser.add_argument('--dueling', dest='dueling', action='store_true', help="Use a Dueling Architecture (DDQN)")
    parser.add_argument('--consecutive_frames', type=int, default=3, help="Number of consecutive frames (action repeat)")
    #
    parser.add_argument('--model_path', type=str, help="Number of training episodes")
    parser.add_argument('--actor_path', type=str, help="Number of training episodes")
    parser.add_argument('--critic_path', type=str, help="Batch size (experience replay)")
    #
    parser.add_argument('--env', type=str, default='DepthFusionBGray-v0',help="OpenAI Gym Environment")
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--epsilon', type=float, default=0.98, help='Epsilon (Explore)')
    parser.add_argument('--epsilon_decay', type=float, default=0.99, help='decay')
    parser.add_argument('--nb_episodes', type=int, default=5000, help="Number of training episodes")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size (experience replay)")
    parser.add_argument('--save_interval', type=int, default=30, help="save_interval")
    #
    parser.set_defaults(render=False)
    return parser.parse_args(args)

def main(args=None):

    # Parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # Check if a GPU ID was set
    # if args.gpu:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # set_session(get_session())
    # with tf.device('/gpu:0'):
    #     config = tf.ConfigProto()
    #     config.gpu_options.allow_growth = True
    #     sess = tf.Session(config=config)
    #     K.set_session(sess)
    #     set_session(sess)
    # Environment Initialization
    # if(args.is_atari):
    #     # Atari Environment Wrapper
    #     env = AtariEnvironment(args)
    #     state_dim = env.get_state_size()
    #     action_dim = env.get_action_size()
    if(args.type=="DDPG"):
        # Continuous Environments Wrapper
        env_before = gym.make(args.env)
        env_unwrapped = env_before.unwrapped
        env = Environment(env_before, args.consecutive_frames)
        # env.reset()

        state_dim = env.get_state_size()
        action_dim = env_unwrapped.action_space.shape[0]
        act_range = env_unwrapped.action_space.high
        print('state: ', state_dim)
        print('action: ', action_dim)
        print('act range', act_range)
    else:
    #     # Standard Environments
    #     env_before = gym.make(args.env)
    #     env = Environment(env_before, args.consecutive_frames)
    #     env.reset()
    #     state_dim = env.get_state_size()
    #     action_dim = env.get_action_size()
        # action_dim = gym.make(args.env).action_space.n
        env_before = gym.make(args.env)
        # env_unwrapped = env_before.unwrapped
        # env_unwrapped.observation_space = env_unwrapped.observation_shape
        # state_dim = env_unwrapped.observation_space.shape
        # action_dim = env_unwrapped.action_space.n
        env = Environment(env_before, args.consecutive_frames)
        # env.reset()

        state_dim = env.get_state_size()
        action_dim = env.get_action_size()
        print('state: ', state_dim)
        print('action: ', action_dim)


    # Pick algorithm to train
    if(args.type=="DDQN"):
        algo = DDQN(action_dim, state_dim, args)
        algo.load_weights(args.model_path)
    elif(args.type=="A2C"):
        algo = A2C(action_dim, state_dim, args.consecutive_frames)
        algo.load_weights(args.actor_path, args.critic_path)
    elif(args.type=="A3C"):
        algo = A3C(action_dim, state_dim, args.consecutive_frames, is_atari=args.is_atari)
        algo.load_weights(args.actor_path, args.critic_path)
    elif(args.type=="DDPG"):
        algo = DDPG(args, action_dim, state_dim, act_range, args.consecutive_frames)
        algo.load_weights(args.actor_path, args.critic_path)

    # Display agent
    old_state, time = env.reset(), 0
    print('old state shape', old_state.shape)
    while True:
       # env.render()
       a = algo.policy_action(old_state)
       if (args.type=="DDPG"):
           a = np.clip(a, -act_range, act_range)
           # print('a', a)
       # print('a', a)
       # print(type(a))
       # print(a.shape)
       old_state, r, done, _ = env.step(a)
       time += 1
       # print('time ',time)
       if done:
           print('done')
           print('Solved in', time, 'steps')
           # break
           old_state = env.reset()
           time = 0
           # break

    env.env.close()

if __name__ == "__main__":
    main()
