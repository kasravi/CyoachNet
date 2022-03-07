import numpy as np
import random

class Environment():
    def __init__(self):
        self.observation_space = ObservationSpace()
        self.shape = self.observation_space.shape

    def seed(self, random_seed):
        self.random_seed = random_seed
        self.action_space = ActionSpace(random_seed)
    
    def reset(self):
        return self.observation_space.initial_state

    def render(self):
        pass

    def step(self, action):
        if action == 7:
            reward = 1
        else:
            reward = -1
        self.observation_space.update()
        i = random.randint(0, 100) 
        return self.observation_space.state, reward, (i>90), i

    def close(self):
        pass    

    # def reshape(self, shape):
    #     return np.reshape(self.observation_space.state, shape)
    
    

class ActionSpace():
    def __init__(self, seed):
        self.n = 10
        self.seed = seed

    def sample(self):
      #random.seed(self.seed) 
      return random.randint(0, self.n)
      


class ObservationSpace():
    def __init__(self):
        self.state = np.random.rand(4,)
        self.shape = self.state.shape
        self.initial_state = self.state
    
    def update(self):
        self.state = np.random.rand(4,)

        




    


    
