import gym
import numpy as np
import math
import random
from tqdm import tqdm
from matplotlib import pyplot as plt

def main():
    env = gym.make("Taxi-v3").env
    results = []
    q_learn = Algorithm(env)

    q_learn.training(0.1, 0.6, 0.1, 5000, 0.001, 1)
    eval = q_learn.evaluate_training(100, 10000)
    test = [0.1, 0.6, 0.1, 5000, 0.001, 1, eval[0], eval[1], eval[2]]
    results.append(test)

    q_learn.training(0.1, 0.6, 0.1, 8000, 0.001, 1)
    eval = q_learn.evaluate_training(100, 10000)
    test = [0.1, 0.6, 0.1, 8000, 0.001, 1, eval[0], eval[1], eval[2]]
    results.append(test)

    q_learn.training(0.1, 0.6, 0.1, 10000, 0.001, 1)
    eval = q_learn.evaluate_training(100, 10000)
    test = [0.1, 0.6, 0.1, 10000, 0.001, 1, eval[0], eval[1], eval[2]]
    results.append(test)

    q_learn.training(0.2, 0.6, 0.1, 8000, 0.001, 1)
    eval = q_learn.evaluate_training(100, 10000)
    test = [0.2, 0.6, 0.1, 8000, 0.001, 1, eval[0], eval[1], eval[2]]
    results.append(test)

    q_learn.training(0.3, 0.6, 0.1, 8000, 0.001, 1)
    eval = q_learn.evaluate_training(100, 10000)
    test = [0.3, 0.6, 0.1, 8000, 0.001, 1, eval[0], eval[1], eval[2]]
    results.append(test)

    q_learn.training(0.5, 0.6, 0.1, 7000, 0.001, 1)
    eval = q_learn.evaluate_training(100, 10000)
    test = [0.5, 0.6, 0.1, 7000, 0.001, 1, eval[0], eval[1], eval[2]]
    results.append(test)

    q_learn.training(0.5, 0.4, 0.1, 5000, 0.001, 1)
    eval = q_learn.evaluate_training(100, 10000)
    test = [0.5, 0.4, 0.1, 5000, 0.001, 1, eval[0], eval[1], eval[2]]
    results.append(test)

    q_learn.training(0.5, 1, 0.1, 5000, 0.001, 1)
    eval = q_learn.evaluate_training(100, 10000)
    test = [0.5, 1, 0.1, 5000, 0.001, 1, eval[0], eval[1], eval[2]]
    results.append(test)

    q_learn.training(0.5, 1, 0.1, 3000, 0.001, 1)
    eval = q_learn.evaluate_training(100, 10000)
    test = [0.5, 1, 0.1, 3000, 0.001, 1, eval[0], eval[1], eval[2]]
    results.append(test)

    q_learn.training(0.5, 1, 0.5, 3000, 0.001, 1)
    eval = q_learn.evaluate_training(100, 10000)
    test = [0.5, 1, 0.5, 3000, 0.001, 1, eval[0], eval[1], eval[2]]
    results.append(test)

    q_learn.training(0.5, 1, 0.5, 1000, 0.001, 1)
    eval = q_learn.evaluate_training(100, 10000)
    test = [0.5, 1, 0.5, 1000, 0.001, 1, eval[0], eval[1], eval[2]]
    results.append(test)

    q_learn.training(0.5, 1, 0.3, 2000, 0.001, 1)
    eval = q_learn.evaluate_training(100, 10000)
    test = [0.5, 1, 0.3, 1000, 0.001, 1, eval[0], eval[1], eval[2]]
    results.append(test)

    q_learn.training(0.5, 1, 0.1, 2000, 0.001, 1)
    eval = q_learn.evaluate_training(100, 10000)
    test = [0.5, 1, 0.1, 1000, 0.001, 1, eval[0], eval[1], eval[2]]
    results.append(test)

    q_learn.training(0.1, 0.6, 0.1, 10000, 0.001, 2)
    eval = q_learn.evaluate_training(100, 10000)
    test = [0.1, 0.6, 0.1, 10000, 0.001, 2, eval[0], eval[1], eval[2]]
    results.append(test)

    q_learn.training(0.1, 0.6, 0.1, 8000, 0.001, 2)
    eval = q_learn.evaluate_training(100, 10000)
    test = [0.1, 0.6, 0.1, 8000, 0.001, 2, eval[0], eval[1], eval[2]]
    results.append(test)

    q_learn.training(0.1, 0.6, 0.1, 5000, 0.001, 2)
    eval = q_learn.evaluate_training(100, 10000)
    test = [0.1, 0.6, 0.1, 5000, 0.001, 2, eval[0], eval[1], eval[2]]
    results.append(test)

    q_learn.training(0.1, 0.6, 0.1, 3000, 0.001, 2)
    eval = q_learn.evaluate_training(100, 10000)
    test = [0.1, 0.6, 0.1, 3000, 0.001, 2, eval[0], eval[1], eval[2]]
    results.append(test)

    q_learn.training(0.1, 0.6, 0.1, 2000, 0.001, 2)
    eval = q_learn.evaluate_training(100, 10000)
    test = [0.1, 0.6, 0.1, 5000, 0.001, 2, eval[0], eval[1], eval[2]]
    results.append(test)

    q_learn.training(0.5, 0.6, 0.1, 2000, 0.001, 2)
    eval = q_learn.evaluate_training(100, 10000)
    test = [0.5, 0.6, 0.1, 2000, 0.001, 2, eval[0], eval[1], eval[2]]
    results.append(test)

    q_learn.training(0.5, 0.6, 0.1, 1500, 0.001, 2)
    eval = q_learn.evaluate_training(100, 10000)
    test = [0.5, 0.6, 0.1, 1500, 0.001, 2, eval[0], eval[1], eval[2]]
    results.append(test)

    q_learn.training(0.5, 0.6, 0.1, 1000, 0.001, 2)
    eval = q_learn.evaluate_training(100, 10000)
    test = [0.5, 0.6, 0.1, 1000, 0.001, 2, eval[0], eval[1], eval[2]]
    results.append(test)

    q_learn.training(0.5, 1, 0.1, 1500, 0.001, 2)
    eval = q_learn.evaluate_training(100, 10000)
    test = [0.5, 1, 0.1, 1500, 0.001, 2, eval[0], eval[1], eval[2]]
    results.append(test)

    q_learn.training(0.5, 0.2, 0.1, 1500, 0.001, 2)
    eval = q_learn.evaluate_training(100, 10000)
    test = [0.5, 0.2, 0.1, 1500, 0.001, 2, eval[0], eval[1], eval[2]]
    results.append(test)

    q_learn.training(0.5, 0.8, 0.1, 1500, 0.001, 2)
    eval = q_learn.evaluate_training(100, 10000)
    test = [0.5, 0.8, 0.1, 1500, 0.001, 2, eval[0], eval[1], eval[2]]
    results.append(test)

    q_learn.training(0.5, 0.6, 1, 1500, 0.001, 2)
    eval = q_learn.evaluate_training(100, 10000)
    test = [0.5, 0.6, 1, 1500, 0.001, 2, eval[0], eval[1], eval[2]]
    results.append(test)

    q_learn.training(0.5, 0.6, 1, 1000, 0.001, 2)
    eval = q_learn.evaluate_training(100, 10000)
    test = [0.5, 0.6, 1, 1000, 0.001, 2, eval[0], eval[1], eval[2]]
    results.append(test)

    q_learn.training(0.5, 0.6, 1, 800, 0.001, 2)
    eval = q_learn.evaluate_training(100, 10000)
    test = [0.5, 0.6, 1, 800, 0.001, 2, eval[0], eval[1], eval[2]]
    results.append(test)

    q_learn.training(0.5, 0.6, 1, 500, 0.001, 2)
    eval = q_learn.evaluate_training(100, 10000)
    test = [0.5, 0.6, 1, 500, 0.001, 2, eval[0], eval[1], eval[2]]
    results.append(test)

    q_learn.training(0.5, 0.6, 3, 500, 0.001, 2)
    eval = q_learn.evaluate_training(100, 10000)
    test = [0.5, 0.6, 3, 500, 0.001, 2, eval[0], eval[1], eval[2]]
    results.append(test)

    q_learn.training(0.5, 0.6, 3, 400, 0.001, 2)
    eval = q_learn.evaluate_training(100, 10000)
    test = [0.5, 0.6, 3, 400, 0.001, 2, eval[0], eval[1], eval[2]]
    results.append(test)

    q_learn.training(0.5, 0.6, 3, 300, 0.001, 2)
    eval = q_learn.evaluate_training(100, 10000)
    test = [0.5, 0.6, 3, 300, 0.001, 2, eval[0], eval[1], eval[2]]
    results.append(test)

    for each in results:
        print(*each,sep=', ')


class Algorithm:
    def __init__(self, environment):
        self.train_iterations = []
        self.train_penalities = []
        self.eval_iterations = None
        self.eval_penalities = None
        self.base_epsilon = None
        self.env = environment
        self.q_table = None

    def greedy_epsilon_strategy(self, epsilon, state):
        if random.uniform(0, 1) < epsilon:
            return self.env.action_space.sample() # Explore action space - takes a random action
        else:
            return np.argmax(self.q_table[state]) # Exploit learned values

    def boltzman_strategy(self, tau, state):
        vector = []
        sum = 0
        for i in range(6):
            sum += math.exp(self.q_table[state,i]/tau)
        for i in range(6):
            vector.append(math.exp(self.q_table[state,i]/tau)/sum)
        actions = [*range(6)]
        return random.choices(actions, weights = vector, k=1)[0]

    def training(self, alpha, gamma, epsilon, epochs, decay, strategy):
        # Hyperparameters
        # alpha - learning rate
        # gama - discount factor
        # epsilon - variable used in greedy algorithm to decide about exploration or exploitation
        # epsilon - variable used as tau in boltzman algorithm
        # epochs - number of episodes through which algorithm will train the agent
        # decay - rate of decay of epsilon variable - faster the decay, faster will algorithm focus on exploitation
        # strategy - whcih policy algorithm should use - 1 for greedy 2 for boltzman
        self.train_iterations.clear()
        self.train_penalities.clear()
        self.base_epsilon = epsilon
        self.q_table =  np.zeros([self.env.observation_space.n, self.env.action_space.n])

        with tqdm(total=epochs) as progress:
            for i in range(epochs):
                state = self.env.reset()

                penalties, reward, = 0, 0
                done = False

                while not done:

                    if strategy == 1:
                        action = self.greedy_epsilon_strategy(epsilon, state)
                        if epsilon - decay >= 0:
                            epsilon -= decay
                        else:
                            epsilon = 0
                    elif strategy == 2:
                        action = self.boltzman_strategy(epsilon, state)

                    next_state, reward, done, info = self.env.step(action)

                    old_value = self.q_table[state, action]
                    next_max = np.max(self.q_table[next_state])

                    new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                    self.q_table[state, action] = new_value

                    if reward == -10:
                        penalties += 1

                    state = next_state

                self.train_iterations.append(i)
                self.train_penalities.append(penalties)
                progress.update(1)

    def evaluate_training(self, episodes, max_steps):
        self.eval_iterations, self.eval_penalities, errors = 0, 0, 0

        for _ in range(episodes):
            state = self.env.reset()
            epochs, penalties, reward = 0, 0, 0

            done = False

            while not done:
                action = np.argmax(self.q_table[state])
                state, reward, done, info = self.env.step(action)

                if reward == -10:
                    penalties += 1

                epochs += 1
                if epochs > max_steps:
                    errors += 1
                    done = True

            self.eval_penalities += penalties
            self.eval_iterations += epochs

        return errors, self.eval_iterations/episodes, self.eval_penalities/episodes


if __name__ == "__main__":
    main()