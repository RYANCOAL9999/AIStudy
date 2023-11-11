import gym
import random
import numpy as np

env = gym.make("Taxi-V3")

env.render()

total_episodes = 50000
total_test_episodes = 100
max_steps = 99

learning_rate = 0.7
gamma = 0.6

#Exploration Parameters
epilson = 1.0
max_epilson = 1.0
min_epilson = 0.01
decay_rate = 0.01

action_size = env.action_space.n

print(
    "Action Size = ", 
    action_size
)

state_size = env.observation_space.n

print(
    "State Size = ", 
    state_size
)

qtable = np.zeros(
    (
        state_size, 
        action_size
    )
)

print(qtable)

for episode in range(total_episodes):
    #reset the environment
    state = env.reset()
    step = 0
    done = False
    action = None

    for step in range(max_steps):
        #choose an action a in the current state
        #we first randomize a number
        exp_exp_tradeoff = random.uniform(0, 1)

        #if this number is greater than epsilson, then we are in the situation of exploitation (taking the biggest Q value for this state)
        if exp_exp_tradeoff > epilson:
            action = np.argmax(
                qtable[
                    state,
                    :
                ]
            )

        #else we will do a random choice, i.e., exploration
        else:
            action = env.action_space.sample()
        
        #Now we are talking the action(a) and moving to the state(s') and getting the reward(r)
        new_state, reward, done, info = env.step(action)

        #We update our Qtable uing the Qlearning equation
        qtable[state, action] = qtable[state, action] + learning_rate * (
                                                                            reward 
                                                                            + gamma * (
                                                                                np.max(
                                                                                    qtable[
                                                                                        new_state, 
                                                                                        :
                                                                                    ]
                                                                                )
                                                                            ) 
                                                                            - qtable[
                                                                                state, 
                                                                                action
                                                                            ]
                                                                        )

        #Updating our state
        state = new_state

        #if done, then finish the episode
        if done == True:
            break
        
    #We reduce the epsilson (as we need less exploration)
    epilson = min_epilson + (
                                max_epilson 
                                - min_epilson
                            ) * np.exp(-decay_rate*episode)

    




