import numpy as np
import matplotlib.pyplot as plt

# Function to perform forward kinematics
def forward_kinematics(theta1, theta2, theta3, L1, L2, L3):
    # Joint positions

    # L1 joint
    x1 = L1 * np.cos(theta1)
    y1 = L1 * np.sin(theta1)

    # L2 joint
    x2 = x1 + L2 * np.cos(theta1 + theta2)
    y2 = y1 + L2 * np.sin(theta1 + theta2)

    # L3 joint
    x3 = x2 + L3 * np.cos(theta1 + theta2 + theta3)
    y3 = y2 + L3 * np.sin(theta1 + theta2 + theta3)


    return [(0, 0), (x1, y1), (x2, y2), (x3, y3)]

# Function to perform inverse kinematics (returns possible solutions)
def inverse_kinematics(x, y, L1, L2, L3):

    # Calculate the wrist position
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    # Check if the point is reachable
    if r > L1 + L2 + L3:
        print("Point is outside the reachable workspace.")
        return None

    # Calculate possible theta2 values (elbow-up and elbow-down)
    cos_theta2 = (r**2 - L1**2 - L2**2) / (2 * L1 * L2)

    # Check if the point is reachable
    if abs(cos_theta2) > 1:
        print("Singularity or point not reachable.")
        return None
    sin_theta2_up = np.sqrt(1 - cos_theta2**2)
    sin_theta2_down = -sin_theta2_up
    theta2_up = np.arctan2(sin_theta2_up, cos_theta2)
    theta2_down = np.arctan2(sin_theta2_down, cos_theta2)

    # Calculate corresponding theta1 and theta3
    def calculate_theta1_theta3(theta2, elbow_up=True):
        k1 = L1 + L2 * np.cos(theta2)
        k2 = L2 * np.sin(theta2)
        theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)
        theta3 = phi - theta1 - theta2
        return theta1, theta2, theta3

    # Return both solutions
    solution_up = calculate_theta1_theta3(theta2_up, elbow_up=True)
    solution_down = calculate_theta1_theta3(theta2_down, elbow_up=False)
    return solution_up, solution_down


# Function to plot the robot arm
def plot_robot_arm(joint_positions, color='blue'):
    x_coords, y_coords = zip(*joint_positions)
    plt.plot(x_coords, y_coords, '-o', color=color)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid(True)
    plt.xlabel('X')
    plt.ylabel('Y')

# Function to visualize the workspace
def workspace(L1, L2, L3):
    # Define the range of theta values
    theta = np.linspace(0, 2 * np.pi, 100)
    for L in [L1 + L2 + L3, L1 + L2, L1]:
        x = L * np.cos(theta)
        y = L * np.sin(theta)
        plt.plot(x, y, 'r--', alpha=0.5)

# Main function
def simulate_robot_arm(L1, L2, L3, targets):
    plt.figure()
    workspace(L1, L2, L3)
    for target in targets:
        x, y = target
        solutions = inverse_kinematics(x, y, L1, L2, L3)
        if solutions:
            for solution in solutions:
                theta1, theta2, theta3 = solution
                joint_positions = forward_kinematics(theta1, theta2, theta3, L1, L2, L3)
                plot_robot_arm(joint_positions)
                plt.pause(1)  # Pause to visualize movement

    plt.show()


# Define the robot arm parameters
L1 = 3
L2 = 2
L3 = 1

# Define target positions
targets = [(3, 3), (4, 2), (2, 4), (1, 1)]

# Simulate the robot arm
simulate_robot_arm(L1, L2, L3, targets)