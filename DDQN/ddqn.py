import sys
import random
import numpy as np

from tqdm import tqdm
from .agent import Agent
from random import random, randrange

from utils.memory_buffer import MemoryBuffer
from utils.networks import tfSummary
from utils.stats import gather_stats

import os

class DDQN:
    """ Deep Q-Learning Main Algorithm
    """

    def __init__(self, action_dim, state_dim, args):
        """ Initialization
        """
        # session = K.get_session()
        # Environment and DDQN parameters
        self.with_per = args.with_per
        self.action_dim = action_dim
        if args.consecutive_frames == 1:
            self.state_dim = state_dim
        else:
            self.state_dim = state_dim+( args.consecutive_frames, )
        print('state dim', self.state_dim)
        # self.state_dim = state_dim
        #
        self.lr = 2.5e-4
        self.gamma = 0.95
        # self.epsilon = 0.8
        # self.epsilon_decay = 0.99
        self.epsilon = args.epsilon
        self.epsilon_decay = args.epsilon_decay
        # self.epsilon_minimum = 0.05
        self.epsilon_minimum = 0.05
        # self.buffer_size = 20000
        self.buffer_size = 10000
        #
        if(len(state_dim) < 3):
            self.tau = 1e-2
        else:
            self.tau = 1.0
        # Create actor and critic networks
        self.agent = Agent(self.state_dim, action_dim, self.lr, self.tau, args.dueling)
        # Memory Buffer for Experience Replay
        self.buffer = MemoryBuffer(self.buffer_size, args.with_per)

        exp_dir = '{}/models/'.format(args.type)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        self.export_path = '{}{}_ENV_{}_NB_EP_{}_BS_{}'.format(exp_dir,
            args.type,
            args.env,
            args.nb_episodes,
            args.batch_size)
        # self.save_interval = 20
        self.save_interval = args.save_interval
        # print('args save interval', self.save_interval)
    def policy_action(self, s):
        """ Apply an espilon-greedy policy to pick next action
        """
        if random() <= self.epsilon:
            # print('random')
            return randrange(self.action_dim)
        else:
            # print('predict')
            return np.argmax(self.agent.predict(s)[0])

    def train_agent(self, batch_size):
        """ Train Q-network on batch sampled from the buffer
        """
        # Sample experience from memory buffer (optionally with PER)
        s, a, r, d, new_s, idx = self.buffer.sample_batch(batch_size)

        # Apply Bellman Equation on batch samples to train our DDQN
        q = self.agent.predict(s)
        # print('predict')
        next_q = self.agent.predict(new_s)
        q_targ = self.agent.target_predict(new_s)

        for i in range(s.shape[0]):
            old_q = q[i, a[i]]
            if d[i]:
                q[i, a[i]] = r[i]
            else:
                next_best_action = np.argmax(next_q[i,:])
                q[i, a[i]] = r[i] + self.gamma * q_targ[i, next_best_action]
            if(self.with_per):
                # Update PER Sum Tree
                self.buffer.update(idx[i], abs(old_q - q[i, a[i]]))
        # Train on batch
        self.agent.fit(s, q)
        # Decay epsilon
        if self.epsilon_decay>self.epsilon_minimum:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_minimum

    def train(self, env, args, summary_writer):
        """ Main DDQN Training Algorithm
        """

        results = []
        tqdm_e = tqdm(range(args.nb_episodes), desc='Score', leave=True, unit=" episodes")

        for e in tqdm_e:
            # Reset episode
            time, cumul_reward, done  = 0, 0, False
            old_state = env.reset()

            while not done:
                if args.render: env.render()
                # Actor picks an action (following the policy)
                a = self.policy_action(old_state)
                # print('action ', a)
                # Retrieve new state, reward, and whether the state is terminal
                new_state, r, done, _ = env.step(a)
                # print('reward', r)
                # Memorize for experience replay
                self.memorize(old_state, a, r, done, new_state)
                # Update current state
                old_state = new_state
                cumul_reward += r
                time += 1
                # Train DDQN and transfer weights to target network
                if(self.buffer.size() > args.batch_size):
                    # print('train agent')
                    self.train_agent(args.batch_size)
                    self.agent.transfer_weights()

            # print('memory buffer:', self.buffer.size())
            # Gather stats every episode for plotting
            if(args.gather_stats):
                mean, stdev = gather_stats(self, env)
                results.append([e, mean, stdev])

            # Export results for Tensorboard
            score = tfSummary('score', cumul_reward)
            summary_writer.add_summary(score, global_step=e)
            summary_writer.flush()

            # Display score
            tqdm_e.set_description("Score: " + str(cumul_reward))
            tqdm_e.refresh()
            # print('e', e)
            if (e%self.save_interval == 0)&(e!=0):
                # print('save')
                self.save_weights(self.export_path,e)

        return results

    def memorize(self, state, action, reward, done, new_state):
        """ Store experience in memory buffer
        """

        if(self.with_per):
            q_val = self.agent.predict(state)
            q_val_t = self.agent.target_predict(new_state)
            next_best_action = np.argmax(q_val)
            new_val = reward + self.gamma * q_val_t[0, next_best_action]
            td_error = abs(new_val - q_val)[0]
        else:
            td_error = 0
        self.buffer.memorize(state, action, reward, done, new_state, td_error)

    def save_weights(self, path, ep = 10000):
        path += '_LR_{}'.format(self.lr)
        path +='_ep_{}'.format(ep)
        if(self.with_per):
            path += '_PER'
        self.agent.save(path)

    def load_weights(self, path):
        self.agent.load_weights(path)
