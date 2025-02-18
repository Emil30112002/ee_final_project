import math
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


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

    # Ensure the new point and the optimal point are within the image bounds
    new_point.x = max(0, min(img_width - 1, new_point.x))
    new_point.y = max(0, min(img_height - 1, new_point.y))

    optimal_point.x = max(0, min(img_width - 1, optimal_point.x))
    optimal_point.y = max(0, min(img_height - 1, optimal_point.y))

    return new_point, optimal_point


def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def rrt_star(start, goal, img, max_iter=2000, delta=10, radius=10):
    img_height, img_width = img.shape
    nodes = [start]

    for _ in range(max_iter):
        random_point = Node(random.randint(0, img_width - 1), random.randint(0, img_height - 1))

        nearest_node = min(nodes, key=lambda n: calculate_distance(n, random_point))
        nearest_node_to_goal_point = min(nodes, key=lambda n: calculate_distance(n, goal))

        new_point, optimal_point = calculate_new_and_optimal_points(nearest_node, random_point, goal, nearest_node_to_goal_point, delta, img_width, img_height)

        if img[new_point.y, new_point.x] == 0:
            continue  # Skip if the new point is in an obstacle

        near_nodes = [node for node in nodes if calculate_distance(node, new_point) <= radius]

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

        for node in near_nodes:
            cost = new_point.cost + calculate_distance(node, new_point)
            if cost < node.cost and img[node.y, node.x] != 0:
                node.parent = new_point
                node.cost = cost

    path = []
    current_node = min(nodes, key=lambda n: calculate_distance(n, goal))
    while current_node:
        path.append((current_node.x, current_node.y))
        current_node = current_node.parent

    return path[::-1]  # Reverse the path to start from the start node


def plot_cv2_path(obstacles, path, start, goal):
    # Create a white canvas with the same shape as obstacles
    image_color = np.ones((*obstacles.shape, 3), dtype=np.uint8) * 255

    # Convert obstacles to BGR color space
    image_color[obstacles == 0] = [0, 0, 0]  # Set obstacles to black

    # Draw path
    path = np.array(path)
    for i in range(len(path) - 1):
        cv2.line(image_color, tuple(path[i]), tuple(path[i + 1]), (0, 255, 0), thickness=2)

    # Draw start and goal points
    cv2.circle(image_color, tuple(start), 5, (255, 0, 0), -1)
    cv2.circle(image_color, tuple(goal), 5, (0, 255, 0), -1)

    # Add labels for start and goal
    cv2.putText(image_color, 'Drone position', tuple(start), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(image_color, 'Target point', tuple(goal), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Save image
    cv2.imwrite('final_path.jpg', image_color)


def plot_path(image, path, start, goal):
    plt.imshow(image, cmap="gray")
    path = np.array(path)
    plt.plot(path[:, 0], path[:, 1], "b-", label='Path')
    plt.plot(start[0], start[1], 'go', label='Drone position')
    plt.plot(goal[0], goal[1], 'ro', label='Target point')
    plt.text(start[0]-65, start[1]+25, 'Drone', color='green', fontsize=12, verticalalignment='bottom')
    plt.text(goal[0], goal[1], 'Target', color='red', fontsize=12, verticalalignment='bottom')

    plt.legend()
    plt.title("Example : Start = (400,400) , End = (100,150)")
    plt.savefig("final_path.jpg")
    plt.close()


def create_temp_goal_point(image_array, img_width):
    goal_row_start = 10
    goal_row_end = 30
    goal_col_start = int(img_width / 2) - 50
    goal_col_end = int(img_width / 2) + 50

    # Search for a white pixel in the specified range
    for row in range(goal_row_start, goal_row_end + 1):
        for col in range(goal_col_start, goal_col_end + 1):
            if image_array[row][col] == 1:  # Checking if the pixel is white
                return row, col
    return


def find_rrt_path(img, start_point, goal_coordinates):
    if start_point is None:
        print("Start point couldn't be located, user is not visible")
        return

    if goal_coordinates is None:
        print("Destination goal isn't detected")
        return

    start_x, start_y = start_point
    start = Node(start_y, start_x)

    goal_x, goal_y = goal_coordinates
    goal = Node(goal_x, goal_y)

    path_result = rrt_star(start, goal, img)
    if len(path_result) < 10:
        print("Path is incomplete!")
        return
    if path_result:
        plot_path(img, path_result, (start.x, start.y), (goal.x, goal.y))
    else:
        print("Path was not found!")


# Load the image into a grayscale image array
img = cv2.imread(r"C:\Users\micha\Downloads\autonomous-drone-final-project-master\autonomous-drone-final-project-master\test_pics\pic_after_mask.jpg", cv2.IMREAD_GRAYSCALE)

# Ensure the image is loaded correctly
if img is None:
    print("Image could not be loaded. Check the file path.")
else:
    # Define start and goal points
    start_point = (400,400)
    goal_point = (100, 150)

    # Find RRT* path
    find_rrt_path(img, start_point, goal_point)

    # Load the saved final path image
    img = cv2.imread('final_path.jpg')

    # Display the final path using matplotlib
    if img is not None:
        # Convert from BGR to RGB for proper matplotlib display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title('Drone Path')
        plt.axis('off')  # Hide the axes
        plt.show()
    else:
        print("Final path image could not be loaded.")
