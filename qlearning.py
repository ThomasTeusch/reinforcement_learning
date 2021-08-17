#!/usr/local/bin/python3

# Import libraries
import gym
import random
import numpy as np

# Function to adjust the hyperparameters alpha, gamma and epsilon via exponential decay
def adapt_param(parameter, adaption_rate, epoch, min_epoch):
    if epoch >= min_epoch:
        parameter = parameter * np.exp(adaption_rate)
    return parameter

# Training procedure
def train():
    # Define hyperparameters
    alpha = 0.2
    alpha_adaption_rate = -0.0001
    gamma_init = 0.6
    gamma_adaption_rate = 0.001
    epsilon = 0.5
    epsilon_adaption_rate = -0.0001

    global q_table
    total_epochs = 25001
    print_frequency = int(total_epochs/6)

    for i in range(1, total_epochs):
        state = env.reset()

        epochs, rewards, n_steps = 0, 0, 0
        done = False

        # Adapt new value for alpha
        # --> Should decrease analog to deep learning 
        alpha = adapt_param(alpha, alpha_adaption_rate,
                            i, total_epochs/3)
                            
        # Adapt new value for epsilon:
        # --> Should decrease since exploitation becomes more and more important
        # and exploration is less important
        epsilon = adapt_param(epsilon, epsilon_adaption_rate,
                              i, total_epochs/3)
       
        gamma = gamma_init
        
        # Start training procedure: success if state is solved in less than 500 steps
        while not done and n_steps < 500:
            # If the random value is smaller than epsilon do exploration (do a random step)
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                # else do exploitation (apply best learned step)
                action = np.argmax(q_table[state])

            # Do the step and save the next state, rewards and if finished
            next_state, rewards, done, _ = env.step(action)

            
            # Adjust gamma
            # --> Should increase since the higher the number of steps is, the more important
            # is the short-term reward
            gamma = adapt_param(gamma, gamma_adaption_rate, i, n_steps / 3)
            
            # Update Q-Table
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            q_table[state, action] = (1 - alpha) * old_value + alpha * (rewards + gamma * next_max)

            state = next_state
            epochs += 1
            n_steps += 1

        # Print results after certain episodes
        if i % print_frequency == 0:
            print(f"Episode: {i}, eps: {epsilon:.3f}, " +
                  f"alpha: {alpha:.3f}, gamma: {gamma:.3f}")

# Evaluation procedure
def eval():
    all_states = len(q_table)
    eval_epochs = 0
    n_steps, failed = 0, 0

    for state_counter in range(all_states):
        state = env.reset()
        epochs, reward = 0, 0
        done = False

        # Try to solve state with trained Q-table
        while not done and n_steps < 500:
            action = np.argmax(q_table[state])
            state, reward, done, info = env.step(action)
            epochs += 1
            # If agent needs more than 500 steps it is considered to be failed
            if epochs >= 500:
                failed += 1
                break

        eval_epochs += epochs
        
    # Print evaluation results
    print(f"Average timesteps: {eval_epochs / (all_states - failed)}")
    print(f"Total number of failes: {failed}")


# Create environment and Q-table
env = gym.make("Taxi-v3").env
q_table = np.zeros([env.observation_space.n, env.action_space.n])

print(10*"-" + "START TRAINING" + 10*"-")
train()
print(10*"-" + "START EVALUATION" + 10*"-")
eval()
