import sys
import gym
import time
import random
import numpy as np
from IPython.display import clear_output
from IPython.display import Markdown, display

def printmd(string):
    display(Markdown(string))

def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        print(f"Episode: {frame['episode']}")
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        time.sleep(0.8)

sys.tracebacklimit = 0

# Init Taxi-V2 Env
env = gym.make("Taxi-v3").env

# Init arbitrary values
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
alpha = 0.7 # Momentum 0.2, Current 0.8 Greedy, 0.2 is to reduce volatility and flip flop
gamma = 0.2 # Learning Rate 0.1 Greediness is 10%
epsilon = 0.4 # explore 10% exploit 90%

all_epochs = []
all_penalties = []
training_memory = []

for i in range(1, 50000):
    state = env.reset()
    # Init Vars
    epochs, penalties, reward, = 0, 0, 0
    done = False
    #training
    while not done:
        if random.uniform(0, 1) < epsilon:
            # Check the action space
            action = env.action_space.sample()                                      # for explore
        else:
            # Check the learned values             
            action = np.argmax(q_table[state])                                      # for exploit
        next_state, reward, done, info = env.step(action)                           # gym generate, the environment is ready                   
        old_value = q_table[state, action]          
        next_max = np.max(q_table[next_state])                                      # take highest from q table for exploit
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)   # Update the new value
        q_table[state, action] = new_value

        if reward == -10:                                                           # penalty for performance evaluation
            penalties += 1
            state = next_state
            epochs += 1

        if i % 100 == 0:
            training_memory.append(q_table.copy())
            clear_output(wait=True)
            print("Episode:", i)
            print("Saved q_table during training:", i)

# Episode: 49900
# Saved q_table during training: 49900
# Training finished.
# [[ 0. 0. 0. 0. 0. 0. ]
# [ -1.24999956 -1.24999782 -1.24999956 -1.24999782 -1.24998912 -10.24999782]
# [ -1.249728 -1.24864 -1.249728 -1.24864 -1.2432 -10.24864 ]
# ...
# [ -1.2432 -1.216 -1.2432 -1.24864 -10.2432 -10.2432 ]
# [ -1.24998912 -1.2499456 -1.24998912 -1.2499456 -10.24998912 -10.24998912]
# [ -0.4 -1.08 -0.4 3. -9.4 -9.4 ]]
print("Training finished.")
print(q_table)

state = 499
# [-1.008 -1.0682761 -1.1004 2.72055 -9.2274 -9.1]
# [-0.40000039 -1.07648283 -0.40000128 3. -9.39958914 -9.39998055]
# [-0.4 -1.08 -0.4 3. -9.4 -9.4 ]
# [-0.4 -1.08 -0.4 3. -9.4 -9.4 ]
print(training_memory[0][state])
print(training_memory[20][state])
print(training_memory[50][state])
print(training_memory[200][state])

state = 77
# [-1.07999095 -1.008 3. -1.08309178 -9.1 -9.18424273]
# [-1.08 -0.4 3. -1.08 -9.4 -9.4 ]
# [-1.08 -0.4 3. -1.08 -9.4 -9.4 ]
# [-1.08 -0.4 3. -1.08 -9.4 -9.4 ]
print(training_memory[0][state])
print(training_memory[20][state])
print(training_memory[50][state])
print(training_memory[200][state])

action_dict = {
    0: "move south" ,
    1: "move north", 
    2: "move east",
    3: "move west",
    4: "pickup passenger",
    5: "dropoff passenger"
}

ENV_STATE = env.reset()
print(env.render(mode='ansi'))
state_memory = [i[ENV_STATE] for i in training_memory]
printmd("For state **{}**".format(ENV_STATE))
for step, i in enumerate(state_memory):
    if step % 200==0:
        choice = np.argmax(i)
        printmd("for episode in {}, q table action is {} and it will ... **{}**".format(step*100, choice, action_dict[choice]))
        print(i)
        print()

total_epochs, total_penalties = 0, 0
episodes = 10 # Try 10 rounds
frames = []

for ep in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    done = False

    while not done:
        action = np.argmax(q_table[state]) # deterministic (exploit), not stochastic (explore), only explore in training
        env, state, reward, done, info = env.step(action) #gym
        if reward == -10:
            penalties += 1

        # Put each rendered frame into dict for animation, gym generated
        frames.append({
            'frame': env.render(mode='ansi'),
            'episode': ep,
            'state': state,
            'action': action,
            'reward': reward
            }
        )
        epochs += 1
    
    total_penalties += penalties
    total_epochs += epochs

# Episode: 9
# Timestep: 123
# State: 475
# Action: 5
# Reward: 20
# Results after 10 episodes:
# Average timesteps per episode: 12.3
# Average penalties per episode: 0.0
print_frames(frames)
print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs /episodes}")
print(f"Average penalties per episode: {total_penalties /episodes}")
