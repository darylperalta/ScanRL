""" Deep RL Algorithms for OpenAI Gym environments
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
    #
    parser.add_argument('--nb_episodes', type=int, default=5000, help="Number of training episodes")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size (experience replay)")
    parser.add_argument('--consecutive_frames', type=int, default=1, help="Number of consecutive frames (action repeat)")
    parser.add_argument('--training_interval', type=int, default=30, help="Network training frequency")
    parser.add_argument('--n_threads', type=int, default=8, help="Number of threads (A3C)")
    #
    parser.add_argument('--gather_stats', dest='gather_stats', action='store_true',help="Compute Average reward per episode (slower)")
    parser.add_argument('--render', dest='render', action='store_true', help="Render environment while training")
    parser.add_argument('--env', type=str, default='DepthFusionBGray-v0',help="OpenAI Gym Environment")
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--epsilon', type=float, default=0.8, help='Epsilon (Explore)')
    parser.add_argument('--epsilon_decay', type=float, default=0.8, help='decay')
    parser.add_argument('--pretrain', action='store_true',help='Path to pretrained')
    parser.add_argument('--weights_path', type=str, default='DDQN/models/path.h5')
    parser.add_argument('--save_interval', type=int, default=30, help='save_interval')
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
    # sess = get_session()
    # set_session(sess)
    # K.set_session(sess)
    # with tf.device('/gpu:0'):
    #     config = tf.ConfigProto()
    #     config.gpu_options.allow_growth = True
    #     sess = tf.Session(config=config)
    #     K.set_session(sess)
    #     set_session(sess)
    summary_writer = tf.summary.FileWriter(args.type + "/tensorboard_" + args.env)

    # Environment Initialization
    if(args.is_atari):
        # Atari Environment Wrapper
        env = AtariEnvironment(args)
        state_dim = env.get_state_size()
        action_dim = env.get_action_size()
    elif(args.type=="DDPG"):
        # Continuous Environments Wrapper
        # env = Environment(gym.make(args.env), args.consecutive_frames)
        # env.reset()
        # state_dim = env.get_state_size()
        # action_space = gym.make(args.env).action_space
        # action_dim = action_space.high.shape[0]
        # act_range = action_space.high

        env_before = gym.make(args.env)
        env_unwrapped = env_before.unwrapped
        # env_unwrapped.observation_space = env_unwrapped.observation_shape
        # state_dim = env_unwrapped.observation_space.shape
        # action_dim = env_unwrapped.action_space.n
        env = Environment(env_before, args.consecutive_frames)
        env.reset()

        state_dim = env.get_state_size()
        # action_space = env.action_space
        # action_dim = env.get_action_size()
        action_dim = env_unwrapped.action_space.shape[0]
        act_range = env_unwrapped.action_space.high
        print('state: ', state_dim)
        print('action: ', action_dim)
        print('act range', act_range)
    else:
        # Standard Environments
        # env = Environment(gym.make(args.env), args.consecutive_frames)
        # env.reset()
        # state_dim = env.get_state_size()
        # action_dim = gym.make(args.env).action_space.n

        #unreal
        env_before = gym.make(args.env)
        # env_unwrapped = env_before.unwrapped
        # env_unwrapped.observation_space = env_unwrapped.observation_shape
        # state_dim = env_unwrapped.observation_space.shape
        # action_dim = env_unwrapped.action_space.n
        env = Environment(env_before, args.consecutive_frames)
        env.reset()

        state_dim = env.get_state_size()
        action_dim = env.get_action_size()
        print('state: ', state_dim)
        print('action: ', action_dim)

        # state_dim= (640,380)
        # action_dim =3

    # Pick algorithm to train
    print('args type: ', args.type)
    if(args.type=="DDQN"):
        algo = DDQN(action_dim, state_dim, args)
    elif(args.type=="A2C"):
        algo = A2C(action_dim, state_dim, args.consecutive_frames)
    elif(args.type=="A3C"):
        algo = A3C(action_dim, state_dim, args.consecutive_frames, is_atari=args.is_atari)
    elif(args.type=="DDPG"):
        algo = DDPG(args, action_dim, state_dim, act_range, args.consecutive_frames)

    if args.pretrain:
        print('pretrain')
        algo.load_weights(args.weights_path)

    # Train
    stats = algo.train(env, args, summary_writer)

    # Export results to CSV
    if(args.gather_stats):
        df = pd.DataFrame(np.array(stats))
        df.to_csv(args.type + "/logs.csv", header=['Episode', 'Mean', 'Stddev'], float_format='%10.5f')

    # Save weights and close environments
    exp_dir = '{}/models/'.format(args.type)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    export_path = '{}{}_ENV_{}_NB_EP_{}_BS_{}'.format(exp_dir,
        args.type,
        args.env,
        args.nb_episodes,
        args.batch_size)

    algo.save_weights(export_path)
    env.env.close()

if __name__ == "__main__":
    main()
