from multiprocessing import Process, Pipe
import numpy as np
import cv2
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import env

COMMAND_RESET     = 0
COMMAND_ACTION    = 1
COMMAND_TERMINATE = 2


def preprocess_frame(observation):
  observation = observation.astype(np.float32)
  resized_observation = cv2.resize(observation, (84, 84))
  resized_observation = resized_observation / 255.0
  return resized_observation

def worker(conn, env_name):
  env = gym_super_mario_bros.make(env_name)
  env.reset()
  conn.send(0)

  while True:
    command, arg = conn.recv()

    if command == COMMAND_RESET:
      obs = env.reset()
      state = preprocess_frame(obs)
      conn.send(state)
    elif command == COMMAND_ACTION:
      reward = 0
      for i in range(4):
        obs, r, terminal, _ = env.step(arg)
        reward += r
        if terminal:
          break
      state = preprocess_frame(obs)
      conn.send([state, reward, terminal])
    elif command == COMMAND_TERMINATE:
      break
    else:
      print("bad command: {}".format(command))
    env.close()
    conn.send(0)
    conn.close()


class GymEnvironment(env.Environment):
  @staticmethod
  def get_action_size(env_name):
    env = gym_super_mario_bros.make(env_name)
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    action_size = env.action_space.n
    env.close()
    return action_size
  
  def __init__(self, env_name):
    env.Environment.__init__(self)

    self.conn, child_conn = Pipe()
    self.proc = Process(target=worker, args=(child_conn, env_name))
    self.proc.start()
    self.conn.recv()
    self.reset()

  def reset(self):
    self.conn.send([COMMAND_RESET, 0])
    self.last_state = self.conn.recv()
    
    self.last_action = 0
    self.last_reward = 0

  def stop(self):
    self.conn.send([COMMAND_TERMINATE, 0])
    ret = self.conn.recv()
    self.conn.close()
    self.proc.join()
    print("smb stopped")

  def process(self, action):
    self.conn.send([COMMAND_ACTION, action])
    state, reward, terminal = self.conn.recv()
    
    pixel_change = self._calc_pixel_change(state, self.last_state)
    self.last_state = state
    self.last_action = action
    self.last_reward = reward
    return state, reward, terminal, pixel_change
