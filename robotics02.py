import numpy as np
import matplotlib.pyplot as plt

# Function to compute distance between two points
def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Function to check for collision with obstacles
def is_collision(x, y, obstacles):
    for obs in obstacles:
        obs_x, obs_y, obs_r = obs
        if distance((x, y), (obs_x, obs_y)) <= obs_r:
            return True
    return False

# Function to compute potential field
def compute_potential_field(x, y, goal, obstacles, repulsive_weight=1.0, attractive_weight=1.0):
    # Attractive potential
    attr_potential = attractive_weight * distance((x, y), goal)
    
    # Repulsive potential (avoid obstacles)
    rep_potential = 0
    for obs in obstacles:
        obs_x, obs_y, obs_r = obs
        d = distance((x, y), (obs_x, obs_y))
        if d <= obs_r:
            return np.inf  # Infinite potential inside an obstacle
        rep_potential += repulsive_weight / (d - obs_r)**2

    return attr_potential + rep_potential

# Function to plan path using gradient descent on the potential field
def path_planning(start, goal, obstacles, max_iters=1000, step_size=0.1):
    path = [start]
    current_pos = start

    for _ in range(max_iters):
        # Compute gradient of the potential field
        grad_x = (compute_potential_field(current_pos[0] + step_size, current_pos[1], goal, obstacles) -
                  compute_potential_field(current_pos[0] - step_size, current_pos[1], goal, obstacles)) / (2 * step_size)
        grad_y = (compute_potential_field(current_pos[0], current_pos[1] + step_size, goal, obstacles) -
                  compute_potential_field(current_pos[0], current_pos[1] - step_size, goal, obstacles)) / (2 * step_size)
        
        # Update position along the negative gradient (downhill)
        next_pos = (current_pos[0] - grad_x * step_size, current_pos[1] - grad_y * step_size)
        
        # Check if next position is in collision
        if is_collision(next_pos[0], next_pos[1], obstacles):
            print("Collision detected. Stopping.")
            break
        
        # Move to the next position
        path.append(next_pos)
        current_pos = next_pos
        
        # Check if goal is reached
        if distance(current_pos, goal) < step_size:
            print("Goal reached!")
            break

    return path

# Function to plot the environment, obstacles, and planned path
def plot_environment(start, goal, obstacles, path):
    plt.figure()
    plt.plot(start[0], start[1], 'go', label='Start')
    plt.plot(goal[0], goal[1], 'ro', label='Goal')
    
    # Plot obstacles
    for obs in obstacles:
        circle = plt.Circle((obs[0], obs[1]), obs[2], color='gray', alpha=0.5)
        plt.gca().add_patch(circle)

    # Plot the path
    if path:
        path_x, path_y = zip(*path)
        plt.plot(path_x, path_y, '-b', label='Path')
    
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid(True)
    plt.legend()
    plt.show()

# Define start and goal positions
start = (0, 0)
goal = (7, 7)

# Define obstacles (x, y, radius)
obstacles = [
    (3, 3, 1.5),
    (6, 4, 2),
    (5, 8, 1)
]

# Perform path planning
path = path_planning(start, goal, obstacles)

# Plot the environment and path
plot_environment(start, goal, obstacles, path)
