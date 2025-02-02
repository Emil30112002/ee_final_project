import math
import random
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ultralytics import YOLO
import os

# ----------- DEPTH ESTIMATION MODEL (MiDaS) ----------------
print("Loading MiDaS model...")
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")  # or "DPT_Hybrid"
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()
transform = midas_transforms.dpt_transform
print("MiDaS model loaded.")

# ----------- YOLO MODEL ------------------------------------
print("Loading YOLO model...")
model_yolo = YOLO('yolov8s.pt')  # or your custom model
model_yolo.conf = 0.4  # confidence threshold
model_yolo.iou = 0.45  # NMS IoU threshold
print("YOLO model loaded.")


# ----------- RRT* Node -------------------------------------
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0


# ----------- HELPER FUNCTIONS ------------------------------
def calculate_distance(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def calculate_new_and_optimal_points(nearest_node, random_point, goal, node_for_goal,
                                     delta, img_w, img_h):
    new_point = Node(nearest_node.x, nearest_node.y)
    optimal_point = Node(node_for_goal.x, node_for_goal.y)
    angle_rand = math.atan2(random_point.y - nearest_node.y, random_point.x - nearest_node.x)
    angle_goal = math.atan2(goal.y - node_for_goal.y, goal.x - node_for_goal.x)
    new_point.x += int(delta * math.cos(angle_rand))
    new_point.y += int(delta * math.sin(angle_rand))
    optimal_point.x += int(delta * math.cos(angle_goal))
    optimal_point.y += int(delta * math.sin(angle_goal))
    new_point.x = max(0, min(img_w - 1, new_point.x))
    new_point.y = max(0, min(img_h - 1, new_point.y))
    optimal_point.x = max(0, min(img_w - 1, optimal_point.x))
    optimal_point.y = max(0, min(img_h - 1, optimal_point.y))
    return new_point, optimal_point


def rrt_star(start, goal, obstacle_img, max_iter=2000, delta=10, radius=15):
    img_h, img_w = obstacle_img.shape
    nodes = [start]
    for _ in range(max_iter):
        rand_pt = Node(random.randint(0, img_w - 1), random.randint(0, img_h - 1))
        nearest_node = min(nodes, key=lambda n: calculate_distance(n, rand_pt))
        node_for_goal = min(nodes, key=lambda n: calculate_distance(n, goal))
        new_point, _ = calculate_new_and_optimal_points(nearest_node, rand_pt,
                                                        goal, node_for_goal,
                                                        delta, img_w, img_h)
        if obstacle_img[new_point.y, new_point.x] == 0:
            continue
        near_nodes = [nd for nd in nodes if calculate_distance(nd, new_point) <= radius]
        min_cost_node = nearest_node
        min_cost = nearest_node.cost + calculate_distance(nearest_node, new_point)
        for nd in near_nodes:
            cost = nd.cost + calculate_distance(nd, new_point)
            if cost < min_cost and obstacle_img[nd.y, nd.x] != 0:
                min_cost_node = nd
                min_cost = cost
        new_point.parent = min_cost_node
        new_point.cost = min_cost
        nodes.append(new_point)
        # Use obstacle_img (not obstacle_map) here:
        for nd in near_nodes:
            cost = new_point.cost + calculate_distance(nd, new_point)
            if cost < nd.cost and obstacle_img[nd.y, nd.x] != 0:
                nd.parent = new_point
                nd.cost = cost
    end_node = min(nodes, key=lambda n: calculate_distance(n, goal))
    path = []
    while end_node:
        path.append((end_node.x, end_node.y))
        end_node = end_node.parent
    return path[::-1]


def estimate_size_from_depth(bbox, depth_map, fx=1000):
    x1, y1, x2, y2 = bbox
    w_px = x2 - x1
    h_px = y2 - y1
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    region = 5
    x_start = max(cx - region, 0)
    x_end = min(cx + region, depth_map.shape[1] - 1)
    y_start = max(cy - region, 0)
    y_end = min(cy + region, depth_map.shape[0] - 1)
    depth_region = depth_map[y_start:y_end, x_start:x_end]
    z_est = float(np.median(depth_region))
    W_m = (w_px * z_est) / fx
    H_m = (h_px * z_est) / fx
    return W_m, H_m, z_est


# ------------ PLOTTING -------------------------------------
def plot_path_2d(obstacle_img, path, start_pt, goal_pt, output_path=None):
    """
    Generates and saves a 2D plot of the RRT* path with consistent scales.
    """
    obstacle_img_flipped = np.flipud(obstacle_img)
    plt.figure(figsize=(8, 6))
    plt.imshow(obstacle_img_flipped, cmap='gray', origin='upper')
    plt.xlim(0, obstacle_img.shape[1])
    plt.ylim(0, obstacle_img.shape[0])
    if path:
        path_arr = np.array(path)
        path_arr[:, 1] = obstacle_img_flipped.shape[0] - path_arr[:, 1]
        plt.plot(path_arr[:, 0], path_arr[:, 1], 'b-', label='RRT Path')
    start_flipped = (start_pt[0], obstacle_img_flipped.shape[0] - start_pt[1])
    goal_flipped = (goal_pt[0], obstacle_img_flipped.shape[0] - goal_pt[1])
    plt.plot(start_flipped[0], start_flipped[1], 'go', label='Start')
    plt.plot(goal_flipped[0], goal_flipped[1], 'ro', label='Goal')
    plt.title("RRT* Path in 2D")
    plt.legend()
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.close()


def plot_path_3d(obstacle_img, path, start_pt, goal_pt, traffic_light_points, output_path=None):
    """
    Generates and saves a 3D plot of the RRT* path with fixed axis limits.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    obs_locs = np.where(obstacle_img == 0)
    obs_y = obs_locs[0]
    obs_x = obs_locs[1]
    obs_z = np.zeros_like(obs_x)
    ax.scatter(obs_x, obs_y, obs_z, c='black', s=1, alpha=0.3, label='Obstacles z=0')
    if path:
        path_3d = [(p[0], p[1], 0) for p in path]
        xs = [p[0] for p in path_3d]
        ys = [p[1] for p in path_3d]
        zs = [p[2] for p in path_3d]
        ax.plot(xs, ys, zs, 'b-', linewidth=2, label='RRT Path')
    ax.scatter(start_pt[0], start_pt[1], 0, c='green', s=40, label='Start')
    ax.scatter(goal_pt[0], goal_pt[1], 0, c='red', s=40, label='Goal')
    if traffic_light_points:
        for (cx, cy, z_val) in traffic_light_points:
            ax.scatter(cx, cy, z_val, c='magenta', marker='^', s=50)
        ax.scatter([], [], [], c='magenta', marker='^', s=50, label='Traffic Light (above ground)')
    ax.set_xlim(0, obstacle_img.shape[1])
    ax.set_ylim(0, obstacle_img.shape[0])
    ax.set_zlim(0, 5)
    ax.invert_yaxis()
    ax.set_xlabel("X (cols)")
    ax.set_ylabel("Y (rows)")
    ax.set_zlabel("Z (depth)")
    ax.set_title("3D RRT* & Objects")
    ax.legend()
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.close()


def draw_bounding_boxes(frame, boxes, confs, cls_idxs, class_names):
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int)
        conf = confs[i]
        cls_id = cls_idxs[i]
        if isinstance(class_names, dict):
            label_str = class_names.get(cls_id, f'class_{cls_id}')
        elif isinstance(class_names, list):
            label_str = class_names[cls_id] if cls_id < len(class_names) else f'class_{cls_id}'
        else:
            label_str = f'class_{cls_id}'
        label = f"{label_str} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - base), (x1 + tw, y1), (0, 255, 0), thickness=cv2.FILLED)
        cv2.putText(frame, label, (x1, y1 - base), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


# ------------ VIDEO COMPILATION FUNCTIONS ---------------------
def create_3d_plots_video(results_dir, output_video_path, fps=2):
    image_files = [f for f in os.listdir(results_dir) if f.startswith("final_path_3d_frame_") and f.endswith(".jpg")]
    if not image_files:
        print("No 3D plot images found to compile into a video.")
        return
    image_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    first_image_path = os.path.join(results_dir, image_files[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        print(f"Failed to read the first 3D plot image: {first_image_path}")
        return
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    for image_file in image_files:
        image_path = os.path.join(results_dir, image_file)
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Failed to read 3D plot image: {image_path}. Skipping.")
            continue
        video_writer.write(frame)
        print(f"Added {image_file} to the 3D video.")
    video_writer.release()
    print(f"3D plots video saved as: {output_video_path}")


def create_2d_plots_video(results_dir, output_video_path, fps=2):
    image_files = [f for f in os.listdir(results_dir) if f.startswith("final_path_2d_frame_") and f.endswith(".jpg")]
    if not image_files:
        print("No 2D plot images found to compile into a video.")
        return
    image_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    first_image_path = os.path.join(results_dir, image_files[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        print(f"Failed to read the first 2D plot image: {first_image_path}")
        return
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    for image_file in image_files:
        image_path = os.path.join(results_dir, image_file)
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Failed to read 2D plot image: {image_path}. Skipping.")
            continue
        video_writer.write(frame)
        print(f"Added {image_file} to the 2D video.")
    video_writer.release()
    print(f"2D plots video saved as: {output_video_path}")


def create_object_detection_real_word_video(results_dir, output_video_path, fps=2):
    """
    Creates a video from images starting with 'frame_used_for_plot_'.
    """
    image_files = [f for f in os.listdir(results_dir) if f.startswith("frame_used_for_plot_") and f.endswith(".jpg")]
    if not image_files:
        print("No frame_used_for_plot images found to compile into a video.")
        return
    image_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    first_image_path = os.path.join(results_dir, image_files[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        print(f"Failed to read the first object detection image: {first_image_path}")
        return
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    for image_file in image_files:
        image_path = os.path.join(results_dir, image_file)
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Failed to read image: {image_path}. Skipping.")
            continue
        video_writer.write(frame)
        print(f"Added {image_file} to object_detection_real_word video.")
    video_writer.release()
    print(f"Object detection real word video saved as: {output_video_path}")


# ------------ MAIN PIPELINE FOR VIDEO ------------------------
def main_video():
    video_path = r"C:\Users\tamer\Downloads\input_video.mp4"  # Update as needed
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video:", video_path)
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {frame_width}x{frame_height}, fps={fps}, total_frames={total_frames}")

    # Process alternate frames (every other frame)
    skip_frames = 2
    print(f"Processing every {skip_frames} frame(s) (i.e. alternate frames).")

    # Annotated video writer
    out_path = "output_with_rrt_path.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_fps = fps / skip_frames if skip_frames > 0 else fps
    out = cv2.VideoWriter(out_path, fourcc, out_fps, (frame_width, frame_height))
    print(f"Annotated video will be saved as: {out_path}")

    # Define dynamic start point and goal (ground-projected)
    initial_start_pt = (360, frame_height - 50)
    start_pt = initial_start_pt
    goal_pt = None
    current_path = []
    path_index = 0

    results_dir = "image_results"
    os.makedirs(results_dir, exist_ok=True)

    frame_idx = 0
    processed_frame_counter = 0

    while True:
        for _ in range(skip_frames - 1):
            ret_skip, _ = cap.read()
            if not ret_skip:
                break
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_idx += skip_frames
        processed_frame_counter += 1
        print(f"\n--- Processing frame at index ~{frame_idx}/{total_frames} ---")

        # YOLO detection
        yolo_results = model_yolo.predict(source=frame_bgr, conf=0.4)
        if len(yolo_results) == 0:
            boxes_xyxy = []
            confs = []
            cls_idxs = []
        else:
            result = yolo_results[0]
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_idxs = result.boxes.cls.cpu().numpy().astype(int)

        # Depth estimation
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        input_batch = transform(image_rgb).to(device)
        with torch.no_grad():
            pred = midas(input_batch)
            pred = torch.nn.functional.interpolate(pred.unsqueeze(1),
                                                   size=image_rgb.shape[:2],
                                                   mode='bicubic',
                                                   align_corners=False).squeeze()
        depth_map = pred.cpu().numpy()
        d_min, d_max = depth_map.min(), depth_map.max()
        depth_map = (depth_map - d_min) / (d_max - d_min + 1e-8)
        depth_map = depth_map * 5.0

        obstacle_map = np.ones((frame_height, frame_width), dtype=np.uint8) * 255
        traffic_light_points = []
        class_names = model_yolo.names

        for i, box in enumerate(boxes_xyxy):
            x1, y1, x2, y2 = box.astype(int)
            conf = confs[i]
            cls_id = cls_idxs[i]
            if isinstance(class_names, dict):
                label_str = class_names.get(cls_id, f'class_{cls_id}')
            elif isinstance(class_names, list):
                label_str = class_names[cls_id] if cls_id < len(class_names) else f'class_{cls_id}'
            else:
                label_str = f'class_{cls_id}'
            w_m, h_m, z_est = estimate_size_from_depth((x1, y1, x2, y2), depth_map, fx=1200)
            print(f"[Frame ~{frame_idx}] {label_str} conf={conf:.2f} => {w_m:.2f}m x {h_m:.2f}m, depth~{z_est:.2f}m")
            if label_str != "traffic light":
                cv2.rectangle(obstacle_map, (x1, y1), (x2, y2), 0, thickness=-1)
            else:
                cx = (x1 + x2) // 2
                cy = y2  # ground projection (bottom of the bounding box)
                traffic_light_points.append((cx, cy, z_est))

        if traffic_light_points:
            def distance_to_start(tl):
                return math.sqrt((tl[0] - start_pt[0]) ** 2 + (tl[1] - start_pt[1]) ** 2)

            selected_tl = min(traffic_light_points, key=distance_to_start)
            ground_goal_pt = (selected_tl[0], selected_tl[1])
            goal_pt = ground_goal_pt
            print(f"Selected ground-projected goal: {goal_pt} with depth {selected_tl[2]:.2f}m")
        else:
            print("No traffic light detected in this frame.")
            goal_pt = None

        if current_path and path_index < len(current_path):
            start_pt = current_path[path_index]
            path_index += 1
            print(f"Updated start_pt to next point on path: {start_pt}")
        else:
            print("No current path or reached end of path. Keeping current start_pt.")

        if goal_pt is not None:
            from_pt = Node(start_pt[0], start_pt[1])
            to_pt = Node(goal_pt[0], goal_pt[1])
            new_path = rrt_star(from_pt, to_pt, obstacle_map, max_iter=2000, delta=10, radius=15)
            if len(new_path) < 2:
                print(f"No valid path found by RRT* at frame ~{frame_idx}!")
                path = []
            else:
                print(f"Path found at frame ~{frame_idx} with {len(new_path)} points.")
                path = new_path
                current_path = path
                path_index = 1
        else:
            path = []
            print("No goal point set; skipping path planning.")

        # Draw YOLO bounding boxes on the frame.
        if boxes_xyxy is not None and len(boxes_xyxy) > 0:
            draw_bounding_boxes(frame_bgr, boxes_xyxy, confs, cls_idxs, class_names)

        # Create a copy of the frame (with bounding boxes) for saving without the RRT* path and without markers.
        frame_for_plot = frame_bgr.copy()

        # Draw the RRT* path on the video frame (frame_bgr) only.
        if path:
            # Draw the blue line segments representing the path on the video frame.
            for p_i in range(1, len(path)):
                cv2.line(frame_bgr, path[p_i - 1], path[p_i], (255, 0, 0), 2)
            # Draw start and goal markers on the video frame only.
            cv2.circle(frame_bgr, start_pt, 5, (0, 255, 0), -1)
            cv2.circle(frame_bgr, goal_pt, 5, (0, 0, 255), -1)
            goal_label = "Goal (Traffic Light Base)"
        else:
            cv2.circle(frame_bgr, start_pt, 5, (0, 255, 0), -1)
            goal_label = "Goal: Not Set"

        frame_output_path = os.path.join(results_dir, f"frame_used_for_plot_{processed_frame_counter}.jpg")
        plot_2d_path = os.path.join(results_dir, f"final_path_2d_frame_{processed_frame_counter}.jpg")
        plot_3d_path = os.path.join(results_dir, f"final_path_3d_frame_{processed_frame_counter}.jpg")

        # Save the frame WITHOUT the drawn RRT* path and WITHOUT start/goal markers.
        cv2.imwrite(frame_output_path, frame_for_plot)
        print(f"Saved annotated frame (without RRT* path or markers): {frame_output_path}")

        if goal_pt is not None:
            plot_path_2d(obstacle_map, path, start_pt, goal_pt, output_path=plot_2d_path)
            print(f"Generated 2D plot: {plot_2d_path}")
        else:
            plot_path_2d(obstacle_map, path, start_pt, start_pt, output_path=plot_2d_path)
            print(f"Generated 2D plot with start point as goal: {plot_2d_path}")

        if goal_pt is not None:
            plot_path_3d(obstacle_map, path, start_pt, goal_pt, traffic_light_points, output_path=plot_3d_path)
            print(f"Generated 3D plot: {plot_3d_path}")
        else:
            plot_path_3d(obstacle_map, path, start_pt, start_pt, traffic_light_points, output_path=plot_3d_path)
            print(f"Generated 3D plot with start point as goal: {plot_3d_path}")

        cv2.putText(frame_bgr, f"Frame ~{frame_idx}/{total_frames}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        if path:
            cv2.putText(frame_bgr, "Path: ON", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame_bgr, "Path: OFF", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame_bgr, goal_label, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        out.write(frame_bgr)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("\nDone! Annotated video saved as:", out_path)

    print("\nCreating a video from the 3D plot images...")
    create_3d_plots_video(results_dir, "final_3d_plots_video.mp4", fps=2)

    print("\nCreating a video from the 2D plot images...")
    create_2d_plots_video(results_dir, "final_2d_plots_video.mp4", fps=2)

    print("\nCreating a video from the object detection real world frames...")
    create_object_detection_real_word_video(results_dir, "object_detection_real_word.mp4", fps=2)


# ------------ MAIN PIPELINE FOR SINGLE IMAGE ----------------
def main_image():
    pass


if __name__ == "__main__":
    main_video()
