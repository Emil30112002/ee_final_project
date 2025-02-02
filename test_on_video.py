import os
import cv2
import random
import math
import numpy as np
from ultralytics import YOLO

###############################################################################
# PARAMETERS
###############################################################################
frame_interval = 30  # Process/write every 30th frame
start_point = (400, 400)  # (x, y) format from user perspective

###############################################################################
# CLASS DEFINITIONS
###############################################################################
class Node:
    """
    Node interprets x as column (col) and y as row.
    So, Node(x=col, y=row).
    """
    def __init__(self, x, y):
        self.x = x  # column index
        self.y = y  # row index
        self.parent = None
        self.cost = 0.0

###############################################################################
# HELPER FUNCTIONS
###############################################################################
def calculate_distance(a, b):
    """
    Euclidean distance between two nodes (interpreted as (col, row)).
    """
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

def calculate_new_and_optimal_points(
        nearest_node, random_point, goal, near_goal_node,
        delta, img_width, img_height
    ):
    """
    Generate two candidate points (new_point, optimal_point) using angles
    to random_point and goal. Bound them inside the image size.
    """
    # Copy from nearest_node
    new_point = Node(nearest_node.x, nearest_node.y)
    optimal_point = Node(nearest_node.x, nearest_node.y)

    # Angles: note y is row, x is col, so we use (dy, dx) in atan2
    angle = math.atan2(
        random_point.y - nearest_node.y,
        random_point.x - nearest_node.x
    )
    angle_goal = math.atan2(
        goal.y - near_goal_node.y,
        goal.x - near_goal_node.x
    )

    # Move 'delta' steps in each direction
    new_point.x += int(delta * math.cos(angle))
    new_point.y += int(delta * math.sin(angle))

    optimal_point.x += int(delta * math.cos(angle_goal))
    optimal_point.y += int(delta * math.sin(angle_goal))

    # Bound them by the image dimensions
    # width = number of columns, height = number of rows
    new_point.x = max(0, min(img_width - 1, new_point.x))
    new_point.y = max(0, min(img_height - 1, new_point.y))

    optimal_point.x = max(0, min(img_width - 1, optimal_point.x))
    optimal_point.y = max(0, min(img_height - 1, optimal_point.y))

    return new_point, optimal_point

def rrt_star(start, goal, img, max_iter=2000, delta=10, radius=10):
    """
    RRT* path planning on a 2D binary image.
    0 = obstacle, >0 = free.
    'start' and 'goal' are Node objects (col, row).
    """
    img_height, img_width = img.shape[:2]  # (rows, cols)
    nodes = [start]

    for _ in range(max_iter):
        # Random row in [0, img_height), random col in [0, img_width)
        rand_row = random.randint(0, img_height - 1)
        rand_col = random.randint(0, img_width - 1)
        random_point = Node(rand_col, rand_row)

        # Nearest nodes to random point and to the actual goal
        nearest_node = min(nodes, key=lambda n: calculate_distance(n, random_point))
        near_goal_node = min(nodes, key=lambda n: calculate_distance(n, goal))

        new_point, _ = calculate_new_and_optimal_points(
            nearest_node, random_point, goal, near_goal_node,
            delta, img_width, img_height
        )

        # If new_point is in obstacle (pixel=0), skip
        if img[new_point.y, new_point.x] == 0:
            continue

        # Find nearby nodes within 'radius'
        near_nodes = [
            node for node in nodes
            if calculate_distance(node, new_point) <= radius
        ]

        # Attach new_point to the node that offers the minimum cost
        min_cost_node = nearest_node
        min_cost = nearest_node.cost + calculate_distance(nearest_node, new_point)

        for node in near_nodes:
            cost = node.cost + calculate_distance(node, new_point)
            if cost < min_cost and img[node.y, node.x] != 0:
                min_cost_node = node
                min_cost = cost

        # Set parent/cost for new_point
        new_point.parent = min_cost_node
        new_point.cost = min_cost
        nodes.append(new_point)

        # Rewire near_nodes if new_point offers a cheaper path
        for node in near_nodes:
            cost = new_point.cost + calculate_distance(node, new_point)
            if cost < node.cost and img[node.y, node.x] != 0:
                node.parent = new_point
                node.cost = cost

    # Backtrack from whichever node is closest to the goal
    path = []
    current_node = min(nodes, key=lambda n: calculate_distance(n, goal))
    while current_node:
        # We'll store path as (row, col) to be consistent with indexing
        path.append((current_node.y, current_node.x))
        current_node = current_node.parent

    return path[::-1]  # reverse: from start to goal

def detect_objects(img):
    """
    Run YOLO segmentation.
    Returns bounding boxes, class IDs, segmentations, scores, and a 'normalized_mask'
    where 0 = obstacle, 255 = free (or vice versa).
    """
    model = YOLO('yolo-weights/yolov8x-seg.pt')  # Update path if needed
    height, width, _ = img.shape

    # YOLO inference
    results = model.predict(source=img.copy(), save=False, save_txt=False, retina_masks=True)
    result = results[0]

    # Build a mask from all detected objects
    masks = result.masks.cpu().data.numpy()  # shape: (num_masks, H, W)
    summed_masks = np.sum(masks, axis=0)
    # Convert to 0/255 (binary)
    normalized_mask = 255 - (summed_masks > 0).astype(np.uint8) * 255

    # Build segmentation polygons
    segmentation_contours_idx = []
    for seg in result.masks.xyn:
        # Each seg is Nx2 normalized, multiply by width/height
        seg[:, 0] *= width
        seg[:, 1] *= height
        segment = np.array(seg, dtype=np.int32)
        segmentation_contours_idx.append(segment)

    # Boxes, classes, scores
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    class_ids = np.array(result.boxes.cls.cpu(), dtype="int")
    scores = np.array(result.boxes.conf.cpu(), dtype="float").round(2)

    return bboxes, class_ids, segmentation_contours_idx, scores, normalized_mask

def set_obstacles_to_black(img, mask):
    """
    Black out obstacles in 'img' where 'mask' > 0.
    """
    img[mask > 0] = [0, 0, 0]
    return img

###############################################################################
# DRAWING FUNCTIONS
###############################################################################
def plot_path(image, path, start, goal):
    """
    Draw the RRT* path on the image using OpenCV.
    'path' is a list of (row, col).
    'start', 'goal' are (x, y) from user perspective -> (col, row).
    """
    if len(path) < 2:
        return image  # not enough points

    # Draw path segments as green lines
    for i in range(1, len(path)):
        # path[i] = (row, col).  OpenCV expects (x, y) = (col, row).
        cv2.line(
            image,
            (path[i-1][1], path[i-1][0]),
            (path[i][1], path[i][0]),
            (0, 255, 0),
            2
        )

    # Draw the start point (blue circle)
    # start is given as (x, y), so (col, row).
    cv2.circle(image, start, 5, (255, 0, 0), -1)
    cv2.putText(
        image, 'User',
        (start[0] - 65, start[1] + 25),  # x-65, y+25
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (255, 0, 0), 1
    )

    # Draw the goal point (red circle)
    cv2.circle(image, goal, 5, (0, 0, 255), -1)
    cv2.putText(
        image, 'Destination',
        (goal[0], goal[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (0, 0, 255), 1
    )
    return image

def find_rrt_path(mask_img, start_pt, goal_pt, output_frame):
    """
    mask_img: 2D image with 0 = obstacle, >0 = free
    start_pt: (x, y) from user perspective
    goal_pt: (x, y) from user perspective
    output_frame: color image for drawing
    """
    if start_pt is None:
        print("Start point is None. Cannot plan path.")
        return output_frame

    if goal_pt is None:
        print("Goal point is None. Cannot plan path.")
        return output_frame

    # Convert user-friendly coords (x, y) -> Node(col=x, row=y)
    start_node = Node(start_pt[0], start_pt[1])
    goal_node = Node(goal_pt[0], goal_pt[1])

    # Run RRT* to get path as list of (row, col)
    path_result = rrt_star(start_node, goal_node, mask_img)

    if len(path_result) < 2:
        print("Path is incomplete or too short!")
        return output_frame

    # Draw path on output_frame
    output_frame = plot_path(
        output_frame,
        path_result,
        start_pt,  # (x, y) for start
        goal_pt    # (x, y) for goal
    )
    return output_frame

def find_and_display_rrt_path(original_frame, start_pt):
    """
    1) Detect objects using YOLO
    2) Fill obstacles in original_frame
    3) Find destination (class_id == 9, for example)
    4) Run RRT* and return frame with path drawn
    """
    try:
        # YOLO detection
        bboxes, class_ids, segs, scores, mask_img = detect_objects(original_frame)

        # Fill obstacles on original_frame (turn them black)
        destination_pt = None  # (x, y)
        for bbox, cls_id, seg, score in zip(bboxes, class_ids, segs, scores):
            # If class_id == 9, treat it as "goal"
            if cls_id == 9:
                min_xy = tuple(np.min(seg, axis=0))  # (x_min, y_min)
                destination_pt = (int(min_xy[0]), int(min_xy[1]))  # as (x, y)
            cv2.fillPoly(original_frame, [seg], (0, 0, 0))

        # Find and draw path on a copy
        frame_with_path = find_rrt_path(
            mask_img,       # used for RRT* (2D: 0=obstacle, 255=free)

            start_pt,       # (x, y)
            destination_pt, # (x, y)
            original_frame.copy()  # color image to draw path on
        )
        return frame_with_path

    except Exception as err:
        print(f"Error in find_and_display_rrt_path: {err}")
        return original_frame

###############################################################################
# MAIN
###############################################################################
def main():
    video_path = '/content/final_test.mp4'  # Update path for your input video
    output_video_path = '/content/output_final_test.mp4'

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open input video.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        # Process only every 'frame_interval'-th frame
        if frame_count % frame_interval == 0:
            try:
                # Make a copy to avoid altering the original
                frame_copy = frame.copy()

                # Find/draw the path
                final_frame = find_and_display_rrt_path(frame_copy, start_point)

                # Write this processed frame 'frame_interval' times so it stays longer
                for _ in range(frame_interval):
                    out.write(final_frame)

            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")

        frame_count += 1

    cap.release()
    out.release()
    print(f"Done! Video saved to: {output_video_path}")

if __name__ == '__main__':
    main()