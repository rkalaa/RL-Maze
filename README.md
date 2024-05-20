# RL-Maze

# Hey, this is my first project I am publishing and first time using GitHub outside of CS50, if you have any questions feel free to let me know!


# Maze Solving with Reinforcement Learning (Q-Learning)

This repository demonstrates a reinforcement learning approach, specifically Q-learning, to solve a simple maze navigation problem. The agent learns an optimal policy for navigating the maze from a starting position to a goal position.

## Project Structure
├── maze.py # Maze class definition (Environment)

├── q_learning.py # QLearningAgent class definition (Agent)

├── train.py # Training and testing logic

└── visualize.py # Functions for visualization

## How it Works

1. **Environment:** The `maze.py` file defines the `Maze` class, which represents the maze environment. This environment provides the agent with states (grid cells) and rewards (positive for reaching the goal, negative for hitting walls or taking steps).

2. **Agent:** The `q_learning.py` file implements the `QLearningAgent` class, the core of our reinforcement learning solution. The agent utilizes a Q-table to represent the value function, which estimates the expected future reward for taking specific actions in different states. The agent learns through:
   - **Exploration:** Trying out random actions to gather information about the environment (using an exploration rate that decays over time).
   - **Exploitation:** Selecting actions that lead to the highest expected reward based on the current Q-table estimates.
   - **Q-table Update:** Using the Bellman equation to update Q-values based on observed rewards and the discounted future rewards of subsequent states.

3. **Training:** The `train.py` file manages the training process. It creates the `Maze` environment, initializes the `QLearningAgent`, and runs the training loop. The agent learns over multiple episodes, accumulating experience and refining its policy.

4. **Visualization:** The `visualize.py` file provides functions for:
   - **Visualizing the Maze:** Displaying the maze environment with the starting and goal locations.
   - **Tracking the Agent's Path:** Animating the agent's movement through the maze during training and testing, allowing us to observe its learning progress.

## Getting Started

1. **Install Dependencies:**
   ```bash
   pip install numpy matplotlib
# Customization
You can customize aspects of the maze environment, rewards, learning rate, exploration parameters, and other settings within the train.py file.

# Ideas For The Future
More Complex Environments: I try to have the agent solve different shaped mazes and add obstacles. 

Advanced RL Algorithms: Experiment with other reinforcement learning algorithms, such as Deep Q-Networks for handling more complex state spaces or SARSA for different update rules.

Applications in my other projects: I will use the Q-Learning technique I implemented in this project to others in the future

   
