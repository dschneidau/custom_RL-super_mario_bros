import numpy as np
from collections import deque


class RunFrame(object):
  def __init__(self, state, reward, action, terminal, pixel_change, last_action, last_reward):
    self.state = state
    self.action = action
    self.reward = np.clip(reward, -1, 1) 
    self.terminal = terminal 
    self.pixel_change = pixel_change
    self.last_action = last_action 
    self.last_reward = np.clip(last_reward, -1, 1) 

  def last_action_reward(self, action_size):
    return ExperienceFrame.concat_action_and_reward(self.last_action, action_size,
                                                    self.last_reward)

  def get_action_reward(self, action_size):
    return ExperienceFrame.concat_action_and_reward(self.action, action_size,
                                                    self.reward)

  @staticmethod
  def concat_action_and_reward(action, action_size, reward):
    action_reward = np.zeros([action_size+1])
    action_reward[action] = 1.0
    action_reward[-1] = float(reward)
    return action_reward
  

class Frames(object):
  def __init__(self, history_size):
    self._history_size = history_size
    self._frames = deque(maxlen=history_size)
    self._zero_reward_indices = deque()
    self._non_zero_reward_indices = deque()
    self._top_frame_index = 0


  def add_frame(self, frame):
    if frame.terminal and len(self._frames) > 0 and self._frames[-1].terminal:
      print("Terminal frames continued.")
      return

    frame_index = self._top_frame_index + len(self._frames)
    was_full = self.is_full()
    self._frames.append(frame)

    if frame_index >= 3:
      if frame.reward == 0:
        self._zero_reward_indices.append(frame_index)
      else:
        self._non_zero_reward_indices.append(frame_index)
    
    if was_full:
      self._top_frame_index += 1

      cut_frame_index = self._top_frame_index + 3
      if len(self._zero_reward_indices) > 0 and \
         self._zero_reward_indices[0] < cut_frame_index:
        self._zero_reward_indices.popleft()
        
      if len(self._non_zero_reward_indices) > 0 and \
         self._non_zero_reward_indices[0] < cut_frame_index:
        self._non_zero_reward_indices.popleft()


  def is_full(self):
    return len(self._frames) >= self._history_size


  def sample_sequence(self, sequence_size):
    start_pos = np.random.randint(0, self._history_size - sequence_size -1)

    if self._frames[start_pos].terminal:
      start_pos += 1

    sampled_frames = []
    
    for i in range(sequence_size):
      frame = self._frames[start_pos+i]
      sampled_frames.append(frame)
      if frame.terminal:
        break
    
    return sampled_frames

  
  def sample_rp_sequence(self):
    if np.random.randint(2) == 0:
      from_zero = True
    else:
      from_zero = False
    
    if len(self._zero_reward_indices) == 0:
      from_zero = False
    elif len(self._non_zero_reward_indices) == 0:
      from_zero = True

    if from_zero:
      index = np.random.randint(len(self._zero_reward_indices))
      end_frame_index = self._zero_reward_indices[index]
    else:
      index = np.random.randint(len(self._non_zero_reward_indices))
      end_frame_index = self._non_zero_reward_indices[index]

    start_frame_index = end_frame_index-3
    raw_start_frame_index = start_frame_index - self._top_frame_index

    sampled_frames = []
    
    for i in range(4):
      frame = self._frames[raw_start_frame_index+i]
      sampled_frames.append(frame)

    return sampled_frames