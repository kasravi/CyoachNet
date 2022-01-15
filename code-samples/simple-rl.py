import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle # to save/load q-table
from matplotlib import style
import time


style.use("ggplot") #setting our style

SIZE = 10
EPISODES = 25000
MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25
epsilon = 0.9
EPS_DECAY = 0.9998
SHOW_EVERY = 3000

start_q_table = None #or filename if we have pre-saved Q-Table

LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_N = 1 # player's key in dictionary
FOOD_N = 2 # food's key in dictionary
ENEMY_N = 3 # enemy's key in dictionary

d = {1: (255, 175, 0), 2: (0, 255, 0), 3: (0, 0, 255)} # assigning colors to each component of environment
                                                       # player: blue
                                                       # food: green
                                                       # enemy: red
# class to create the environment
class Blob:
    def __init__(self):
        #initializing the blobs randomly
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)
    
    def __str__(self):
        return f"{self.x}, {self.y}"
    
    ### Creating an observation from our environemnt
    def __sub__(self, other): 
        '''
            Passing delta of x and y for the food and enemy to our agent
        '''
        return (self.x - other.x, self.y - other.y) 

    def action(self, choice):
        '''
            Creating 4 movements option (0, 1, 2, 3) --- we are going to move diagonally!
        '''
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)

    def move (self, x=False, y=False):
        if not x:
            self.x += np.random.randint(-1, 2)

        else:
            self.x += x

        if not y:
            self.y += np.random.randint(-1, 2)

        else:
            self.y += y
        '''
            The next set of conditions are taking care of the situations where we are out of bounds of the environment!
        '''
        if self.x < 0:
            self.x = 0
        
        elif self.x > SIZE - 1:
            self.x = SIZE - 1

        if self.y < 0:
            self.y = 0
        
        elif self.y > SIZE - 1:
            self.y = SIZE - 1


if start_q_table is None:
    q_table = {}
    for x1 in range(-SIZE+1, SIZE):
        for y1 in range(-SIZE+1, SIZE):
            for x2 in range(-SIZE+1, SIZE):
                for y2 in range(-SIZE+1, SIZE):
                    q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-5, 0) for i in range(4)]

else: # In case we have a saved Q-Table to use
    with open(start_q_table, "rb") as f: 
        q_table = pickle.load(f)

episode_rewards = []
for episode in range(EPISODES):
    player = Blob()
    food = Blob()
    enemy = Blob()

    if episode % SHOW_EVERY == 0:
        '''
            To visualize only every 3000 episodes
        '''
        print(f"on # {episode}, epsolon: {epsilon}")
        print(f"{SHOW_EVERY} ep mean {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    episode_reward = 0
    for i in range(200):
        obs = (player-food, player-enemy)
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, 4)

        player.action(action)
        '''
            we may decide to move the food and enemy as well
        '''
        #food.move()
        #enemy.move()

        #Updating the reward
        if player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PENALTY
        elif player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
        else:
            reward = -MOVE_PENALTY
        
        new_obs = (player-food, player-enemy)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]

        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        elif reward == -ENEMY_PENALTY:
            new_q = -ENEMY_PENALTY
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        
        q_table[obs][action] = new_q

        if show:
            '''
                Visualizing the environment and movements
            '''
            env = np.zeros((SIZE, SIZE, 3), dtype= np.uint8) # creates an RBG image of the size of the environment
            env[food.x][food.y] = d[FOOD_N] # Set the food location tile to green
            env[player.x][player.y] = d[PLAYER_N] # Set the player location tile to blue
            env[enemy.x][enemy.y] = d[ENEMY_N] # Set the enemy location tile to redr

            img = Image.fromarray(env, 'RGB')
            img = img.resize((300, 300), resample=Image.BOX)
            cv2.imshow("", np.array(img))
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
                if cv2.waitKey(500) & 0xFF == ord("q"):
                    break
                else:
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        
        episode_reward += reward
        if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
            break
    
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY, )) / SHOW_EVERY, mode="valid")
plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()

with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)
