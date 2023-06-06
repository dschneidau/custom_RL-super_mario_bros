import numpy as np
import unittest
import gym_super_mario_bros

class Environment(object):
    action_size = -1

    @staticmethod
    def create_environment(env_type, env_name):
        import super_mario_gym
        return super_mario_gym.GymEnvironment(env_name)

    @staticmethod
    def get_action_size(env_type, env_name):
        if Environment.action_size >= 0:
          return Environment.action_size

        import super_mario_gym
        Environment.action_size = super_mario_gym.GymEnvironment.get_action_size(env_name)
        return Environment.action_size

    def __init__(self):
        pass

    def process(self, action):
        pass

    def reset(self):
        pass

    def stop(self):
        pass  

    def _subsample(self, a, average_width):
        s = a.shape
        sh = s[0]//average_width, average_width, s[1]//average_width, average_width
        return a.reshape(sh).mean(-1).mean(1)  

    def _calc_pixel_change(self, state, last_state):
        d = np.absolute(state[2:-2,2:-2,:] - last_state[2:-2,2:-2,:])
        # (80,80,3)
        m = np.mean(d, 2)
        c = self._subsample(m, 4)
        return c

        