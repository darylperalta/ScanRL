import gym
import numpy as np
from collections import deque
import cv2
# class Environment(object):
#     """ Environment Helper Class (Multiple State Buffer) for Continuous Action Environments
#     (MountainCarContinuous-v0, LunarLanderContinuous-v2, etc..), and MujuCo Environments
#     """
#     def __init__(self, gym_env, action_repeat):
#         self.env = gym_env
#         self.timespan = action_repeat
#         self.gym_actions = 2 #range(gym_env.action_space.n)
#         self.state_buffer = deque()
#
#     def get_action_size(self):
#         return self.env.action_space.n
#
#     def get_state_size(self):
#         env = self.env.unwrapped
#         self.env.observation_space = env.observation_shape
#         return self.env.observation_space.shape
#
#     def reset(self):
#         """ Resets the game, clears the state buffer.
#         """
#         # Clear the state buffer
#         self.state_buffer = deque()
#         x_t = self.env.reset()
#         s_t = np.stack([x_t for i in range(self.timespan)], axis=0)
#         for i in range(self.timespan-1):
#             self.state_buffer.append(x_t)
#         return s_t
#
#     def step(self, action):
#         x_t1, r_t, terminal, info = self.env.step(action)
#         previous_states = np.array(self.state_buffer)
#         s_t1 = np.empty((self.timespan, *self.env.observation_space.shape))
#         s_t1[:self.timespan-1, :] = previous_states
#         s_t1[self.timespan-1] = x_t1
#         # Pop the oldest frame, add the current frame to the queue
#         self.state_buffer.popleft()
#         self.state_buffer.append(x_t1)
#         return s_t1, r_t, terminal, info
#
#     def render(self):
#         return self.env.render()

class Environment(object):
    """ Environment Helper Class (Multiple State Buffer) for Continuous Action Environments
    (MountainCarContinuous-v0, LunarLanderContinuous-v2, etc..), and MujuCo Environments
    """
    def __init__(self, gym_env, action_repeat):
        self.env = gym_env
        self.timespan = action_repeat
        # self.gym_actions = 2 #range(gym_env.action_space.n)
        self.state_buffer = deque()
        self.normalize = True

    def get_action_size(self):
        return self.env.action_space.n

    def get_state_size(self):
        env = self.env.unwrapped
        self.env.observation_space = env.observation_shape
        # return self.env.observation_space.shape
        return self.env.observation_space.shape[0:2]

    def reset(self):
        """ Resets the game, clears the state buffer.
        """
        # Clear the state buffer
        self.state_buffer = deque()
        x_t = self.env.reset()
        if self.normalize == True:
            x_t = x_t/255.0
        # print('x_t shape reset', x_t.shape)

        s_t = np.stack([x_t for i in range(self.timespan)], axis=2)
        # print('s_t shape reset', s_t.shape)
        for i in range(self.timespan-1):
            self.state_buffer.append(x_t)
            # cv2.imshow('state img reset', x_t)
            # cv2.waitKey(0)
            # cv2.imshow('state img buffer', self.state_buffer[i])
            # cv2.waitKey(0)

        # print('state_buffer shape reset', )
        # print(type(self.state_buffer))
        # print(s_t.shape)
        # s_t = self.env.reset()
        return s_t

    def step(self, action):
        # print('step')

        x_t1, r_t, terminal, info = self.env.step(action)
        if self.normalize == True:
            x_t1 = x_t1/255.0
        # cv2.imshow('state img step', x_t1)
        # cv2.waitKey(0)

        # print('x_t step ', x_t1.shape)
        # previous_states = np.array(self.state_buffer)
        previous_states = np.empty((*self.env.observation_space.shape[0:2], self.timespan-1), dtype = np.float64)
        for i in range(self.timespan-1):
            # filename = 'state img step prev%d' %i
            previous_states[:,:,i]=self.state_buffer[i]
            # cv2.imshow(filename, self.state_buffer[i])
            # cv2.waitKey(0)
            # print(self.state_buffer[i])
            # print(type(self.state_buffer[i]))
            # print(self.state_buffer[i].dtype)

        # print('prev')
        # for i in range(self.timespan-1):
        #     filename = 'state img step prev_%d' %i
        #     # previous_states[:,:,i]=self.state_buffer[i]
        #     cv2.imshow(filename, previous_states[:,:,i])
        #     cv2.waitKey(0)
        #     print(previous_states[:,:,i])
        #     # print(type(previous_states[:,:,i]))
        #     print('type prev state', previous_states.dtype)


        # print('shape previous states', previous_states.shape)
        s_t1 = np.empty((*self.env.observation_space.shape[0:2], self.timespan))
        # print('type', s_t1.dtype)
        # print('shape ', s_t1.shape)
        s_t1[:,:,:self.timespan-1] = previous_states
        s_t1[:,:,self.timespan-1] = x_t1

        # for i in range(self.timespan-1):
        #     # previous_states[:,:,i]=self.state_buffer[i]
        #     filename = 'state img step prev%d' %i
        #     # print('filename', filename)
        #     cv2.imshow(filename,previous_states[:,:,i])
            # cv2.imshow(filename, self.state_buffer[i])
            # cv2.waitKey(0)

        # for i in range(self.timespan):
        #     filename = 'state img actual%d' %i
        #     cv2.imshow(filename,s_t1[:,:,i])
        #     cv2.waitKey(0)

        # # Pop the oldest frame, add the current frame to the queue]
        self.state_buffer.popleft()
        self.state_buffer.append(x_t1)
        # print('shape ', s_t1.shape)

        # s_t1, r_t, terminal, info = self.env.step(action)

        return s_t1, r_t, terminal, info

    def render(self):
        return self.env.render()
