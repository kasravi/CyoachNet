import gym
import numpy as np
import matplotlib.pyplot as plt

#Setting up the environment
env = gym.make("MountainCar-v0")


LEARNING_RATE = 0.1
DISCOUNT = 0.95 #Something like a weight, measure of feauter reward vs current reward
SHOW_EVERY = 2000
EPISODES = 25000


DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

#Exploration settings
epsilon = 1 #not a constant, will decay
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)


#creating the Q-table
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int)) #Use this to look up the 3 Q values for the available actions

for episode in range(EPISODES):
    discrete_state = get_discrete_state(env.reset())
    done = False

    if episode % SHOW_EVERY == 0:
        render = True
        print(episode)
    else:
        render = False



    while not done:

        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(q_table[discrete_state]) 
        else:
            # Get a random action
            action = np.random.randint(0, env.action_space.n)
        
        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)

        if episode % SHOW_EVERY == 0:
            env.render()

        if not done:
            # Maximum possible Q value in the next step (for new state)
            max_future_q = np.max(q_table[new_discrete_state])

            # Current Q value (for current state and action performed)
            current_q = q_table[discrete_state + (action,)]

            # Equation for a new Q value for current state and action
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            # Update Q table with new Q value
            q_table[discrete_state + (action,)] = new_q

        # If goal position is achieved - Update Q value with reward directly
        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action,)] = 0
    
        discrete_state = new_state

    # Decaying is being done in every episode if episode number is with decaying range
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value


env.close()



