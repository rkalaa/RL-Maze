import numpy as np
import matplotlib.pyplot as plt
from maze import Maze
from q_learning import QLearningAgent
from visualize import finish_episode, test_agent, train_agent

# Make the maze layout
maze_layout = np.array([
    [0, 1, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0],
    [1, 1, 0, 1, 1],
    [0, 0, 0, 0, 0]
])

# Make the maze
maze = Maze(maze_layout, (0, 0), (4, 4))

# Show the maze on the screen
maze.show_maze()

# Define what actions the agent can take
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Define how much reward the agent gets
goal_reward = 100
wall_penalty = -10
step_penalty = -1

print("Maze and reward success")

# Create the agent and train
agent = QLearningAgent(maze, maze.maze_height, maze.maze_width)
train_agent(agent, maze, actions, goal_reward, wall_penalty, step_penalty, num_episodes=100)

# Test the trained agent
test_agent(agent, maze, actions, goal_reward, wall_penalty, step_penalty)