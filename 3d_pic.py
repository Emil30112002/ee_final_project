import math
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0

def calculate_new_and_optimal_points(nearest_node, random_point, goal, nearest_node_to_goal_point, delta, img_width, img_height):
    new_point = Node(nearest_node.x, nearest_node.y)
    optimal_point = Node(nearest_node.x, nearest_node.y)

    angle = math.atan2(random_point.y - nearest_node.y, random_point.x - nearest_node.x)
    angle_goal = math.atan2(goal.y - nearest_node_to_goal_point.y, goal.x - nearest_node_to_goal_point.x)

    new_point.x += int(delta * math.cos(angle))
    new_point.y += int(delta * math.sin(angle))

    optimal_point.x += int(delta * math.cos(angle_goal))
    optimal_point.y += int(delta * math.sin(angle_goal))

    # Ensure points are within image bounds
    new_point.x = max(0, min(img_width - 1, new_point.x))
    new_point.y = max(0, min(img_height - 1, new_point.y))

    optimal_point.x = max(0, min(img_width - 1, optimal_point.x))
    optimal_point.y = max(0, min(img_height - 1, optimal_point.y))

    return new_point, optimal_point

def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def rrt_star(start, goal, img, max_iter=2000, delta=10, radius=10):
    """
    RRT* in 2D on a grayscale image:
    - 255 (white) = free
    - 0   (black) = obstacle
    """
    img_height, img_width = img.shape
    nodes = [start]

    for _ in range(max_iter):
        random_point = Node(random.randint(0, img_width - 1), random.randint(0, img_height - 1))

        nearest_node = min(nodes, key=lambda n: calculate_distance(n, random_point))
        nearest_node_to_goal_point = min(nodes, key=lambda n: calculate_distance(n, goal))

        new_point, _ = calculate_new_and_optimal_points(
            nearest_node,
            random_point,
            goal,
            nearest_node_to_goal_point,
            delta,
            img_width,
            img_height
        )

        # Skip if new point is in obstacle
        if img[new_point.y, new_point.x] == 0:
            continue

        # Find all nodes within 'radius'
        near_nodes = [node for node in nodes if calculate_distance(node, new_point) <= radius]

        # Choose the parent with minimal cost
        min_cost_node = nearest_node
        min_cost = nearest_node.cost + calculate_distance(nearest_node, new_point)

        for node in near_nodes:
            cost = node.cost + calculate_distance(node, new_point)
            if cost < min_cost and img[node.y, node.x] != 0:
                min_cost_node = node
                min_cost = cost

        new_point.parent = min_cost_node
        new_point.cost = min_cost

        nodes.append(new_point)

        # Rewire near nodes if it yields a lower cost
        for node in near_nodes:
            cost = new_point.cost + calculate_distance(node, new_point)
            if cost < node.cost and img[node.y, node.x] != 0:
                node.parent = new_point
                node.cost = cost

    # Build path by tracing from the node closest to 'goal'
    path = []
    current_node = min(nodes, key=lambda n: calculate_distance(n, goal))
    while current_node:
        path.append((current_node.x, current_node.y))
        current_node = current_node.parent

    return path[::-1]

def plot_path_2d(image, path, start, goal):
    """
    Basic 2D path plotting with Matplotlib
    """
    plt.imshow(image, cmap="gray")
    path = np.array(path)
    plt.plot(path[:, 0], path[:, 1], 'b-', label='Path')
    plt.plot(start[0], start[1], 'go', label='Drone position')
    plt.plot(goal[0], goal[1], 'ro', label='Target point')

    plt.text(start[0] - 65, start[1] + 25, 'Drone', color='green', fontsize=12,
             verticalalignment='bottom')
    plt.text(goal[0], goal[1], 'Target', color='red', fontsize=12,
             verticalalignment='bottom')

    plt.title("RRT* Path in 2D")
    plt.legend()
    plt.savefig("final_path.jpg")
    plt.close()

def plot_path_3d(path, start, goal, obstacles_img):
    """
    Plots obstacles and path on the same plane (z=0) in a 3D plot.
    - Obstacles (black pixels) => scattered on z=0
    - Path => line on z=0
    - Start/Goal => points on z=0
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 1) Plot obstacles at z=0
    #    We'll collect (x,y) for every black pixel, i.e. obstacles_img == 0
    obs_indices = np.where(obstacles_img == 0)
    obs_y = obs_indices[0]  # row
    obs_x = obs_indices[1]  # col
    obs_z = np.zeros_like(obs_x)  # all zero for z

    ax.scatter(obs_x, obs_y, obs_z, c='black', s=1, alpha=0.3, label='Obstacles')

    # 2) Plot the path on z=0
    path_3d = [(p[0], p[1], 0) for p in path]  # keep z=0
    xs = [pt[0] for pt in path_3d]
    ys = [pt[1] for pt in path_3d]
    zs = [pt[2] for pt in path_3d]

    ax.plot(xs, ys, zs, 'b-', linewidth=2, label='Drone Path')

    # 3) Mark Start and Goal
    #    Also at z=0
    ax.scatter(start[0], start[1], 0, color='green', s=50, label='Start')
    ax.scatter(goal[0], goal[1], 0, color='red', s=50, label='Goal')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("3D View (Everything on z=0)")
    ax.legend()

    plt.savefig("final_path_3d.jpg")
    plt.show()
    plt.close()

def find_rrt_path(img, start_point, goal_coordinates):
    """
    Main function:
    - Runs RRT*
    - Plots path in 2D
    - Plots everything on z=0 in 3D
    """
    if start_point is None:
        print("Start point couldn't be located, user is not visible")
        return
    if goal_coordinates is None:
        print("Destination goal isn't detected")
        return

    start_x, start_y = start_point
    start = Node(start_y, start_x)

    goal_x, goal_y = goal_coordinates
    goal = Node(goal_y, goal_x)

    path_result = rrt_star(start, goal, img)

    if len(path_result) < 2:
        print("Path is incomplete or not found!")
        return

    # 1) Plot 2D Path
    plot_path_2d(img, path_result, (start.x, start.y), (goal.x, goal.y))

    # 2) Plot 3D Visualization (z=0)
    plot_path_3d(path_result, (start.x, start.y), (goal.x, goal.y), img)

if __name__ == "__main__":
    img_path = r'D:\autonomous-drone-final-project-master\autonomous-drone-final-project-master\test_pics\pic_after_mask.jpg'
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Image could not be loaded. Check the path.")
    else:
        # Example start & goal
        start_point = (400, 400)  # (x, y)
        goal_point = (100, 150)   # (x, y)

        find_rrt_path(img, start_point, goal_point)

        # Optionally display final_path.jpg
        final_path_img = cv2.imread("final_path.jpg")
        if final_path_img is not None:
            final_path_img_rgb = cv2.cvtColor(final_path_img, cv2.COLOR_BGR2RGB)
            plt.imshow(final_path_img_rgb)
            plt.title('Drone Path (2D)')
            plt.axis('off')
            plt.show()
        else:
            print("Could not load final_path.jpg")
