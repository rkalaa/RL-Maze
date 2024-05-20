import numpy as np
import matplotlib.pyplot as plt

def finish_episode(agent, maze, episode, actions, goal_reward, wall_penalty, step_penalty, train=True):
    # Let the agent explore the maze
    current_state = maze.start_position  # Use the correct property name
    is_done = False
    total_reward = 0
    steps = 0
    path = [current_state]

    # Only show the agent moving if it's every 10th episode
    show_animation = episode % 10 == 0

    # Make the animation window
    if show_animation:
        plt.figure(figsize=(5, 5))
        plt.imshow(maze.maze, cmap='gray')
        plt.text(maze.start_position[0], maze.start_position[1], 'S', ha='center', va='center', color='red', fontsize=20)
        plt.text(maze.goal_position[0], maze.goal_position[1], 'G', ha='center', va='center', color='green', fontsize=20)
        plt.xticks([]), plt.yticks([])

        plt.ion()  # Enable interactive mode
        plt.show() # show the plot

    # Keep exploring until the agent reaches the goal
    while not is_done:
        # Choose an action
        action = agent.get_action(current_state, episode)
        # Move to the next state
        next_state = (current_state[0] + actions[action][0], current_state[1] + actions[action][1])

        # Check if the agent hit a wall
        if next_state[0] < 0 or next_state[0] >= maze.maze_height or next_state[1] < 0 or next_state[1] >= maze.maze_width or maze.maze[next_state[1]][next_state[0]] == 1:
            reward = wall_penalty
            next_state = current_state
        # Check if the agent reached the goal
        elif next_state == maze.goal_position:  # Use the correct property name
            path.append(current_state)
            reward = goal_reward
            is_done = True
        else:
            path.append(current_state)
            reward = step_penalty

        # Keep track of the total reward and steps
        total_reward += reward
        steps += 1

        # Learn from the experience if it's training time
        if train == True:
            agent.update_q_table(current_state, action, next_state, reward)

        # Show the agent moving
        if show_animation:
            plt.text(current_state[0], current_state[1], "#", va='center', color='blue', fontsize=20)
            plt.pause(0.1)  # Wait a little bit
            plt.text(current_state[0], current_state[1], " ", va='center', color='blue', fontsize=20) # Clear the agent marker

        # Update the current state
        current_state = next_state

    if show_animation:
        plt.ioff() # turn off interactive mode
        plt.show()

    # Return the results
    return total_reward, steps, path

print("The finish_episode function is ready!")

def test_agent(agent, maze, actions, goal_reward, wall_penalty, step_penalty, num_episodes=1):
    # Test the agent!
    total_reward, steps, path = finish_episode(agent, maze, num_episodes, actions, goal_reward, wall_penalty, step_penalty, train=False)

    print("The agent found this path:")
    for row, col in path:
        print(f"({row}, {col})-> ", end='')
    print("Goal!")

    print("It took", steps, "steps.")
    print("It got a total reward of", total_reward)

    if plt.gcf().get_axes():
        plt.cla()

    plt.figure(figsize=(5,5))
    plt.imshow(maze.maze, cmap='gray')
    plt.text(maze.start_position[0], maze.start_position[1], 'S', ha='center', va='center', color='red', fontsize=20)
    plt.text(maze.goal_position[0], maze.goal_position[1], 'G', ha='center', va='center', color='green', fontsize=20)

    for position in path:
        plt.text(position[0], position[1], "#", va='center', color='blue', fontsize=20)

    plt.xticks([]), plt.yticks([])
    plt.grid(color='black', linewidth=2)
    plt.show()

    return steps, total_reward

def train_agent(agent, maze, actions, goal_reward, wall_penalty, step_penalty, num_episodes=100):
    # Train the agent!
    total_rewards = []
    total_steps = []

    for episode in range(num_episodes):
        total_reward, steps, path = finish_episode(agent, maze, episode, actions, goal_reward, wall_penalty, step_penalty, train=True)
        total_rewards.append(total_reward)
        total_steps.append(steps)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(total_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward per Episode')

    average_reward = sum(total_rewards) / len(total_rewards)
    print(f"The average reward is: {average_reward}")

    plt.subplot(1, 2, 2)
    plt.plot(total_steps)
    plt.xlabel('Episode')
    plt.ylabel('Steps Taken')
    plt.ylim(0, 100)
    plt.title('Steps per Episode')

    average_steps = sum(total_steps) / len(total_steps)
    print(f"The average steps is: {average_steps}")

    plt.tight_layout()
    plt.show()

print("The train_agent function is ready!")