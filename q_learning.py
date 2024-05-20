import numpy as np

# Class for agent
class QLearningAgent:
    def __init__(self, maze, maze_height, maze_width, learning_rate=0.1, discount=0.9, explore_start=1.0, explore_end=0.01, num_episodes=100):
        # Table to keep track of values
        self.q_table = np.zeros((maze_height, maze_width, 4))
        self.learning_rate = learning_rate
        self.discount = discount
        self.explore_start = explore_start
        self.explore_end = explore_end
        self.num_episodes = num_episodes
        self.maze = maze
    
    def get_explore_rate(self, episode):
        # Figure out how much to explore based on the episode number
        explore_rate = self.explore_start * (self.explore_end / self.explore_start) ** (episode / self.num_episodes)
        return explore_rate

    def get_action(self, state, episode):
        # Decide what action to take
        explore_rate = self.get_explore_rate(episode)
        if np.random.rand() < explore_rate:
            # Explore randomly
            return np.random.randint(4)
        else:
            # Choose the best action based on the Q-table
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, next_state, reward):
        # Learn from the experience
        best_next_action = np.argmax(self.q_table[next_state])
        current_q_value = self.q_table[state][action]
        new_q_value = current_q_value + self.learning_rate * (reward + self.discount * self.q_table[next_state][best_next_action] - current_q_value)
        self.q_table[state][action] = new_q_value

print("QLearn success")