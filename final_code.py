import cv2
import time
import random
import math
import threading
import matplotlib
import numpy as np
from djitellopy import Tello
from ultralytics import YOLO

# Force matplotlib to use TkAgg backend
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# ----------------------------
# Configuration
# ----------------------------
TARGET_CLASS = "person"
FRAME_SIZE = (480, 640)  # (height, width)

# Pathfinding parameters
RRT_STEP = 40
RRT_MAX_ITER = 1000
OBSTACLE_MARGIN = 20
GOAL_RADIUS = 30

# Control parameters
FORWARD_GAIN = 30  # Reduced for smoother movement
LATERAL_GAIN = 25  # Reduced for smoother movement
YAW_GAIN = 25
MIN_MOVE = 1

# Safety parameters
MIN_DISTANCE = 50
SAFE_HEIGHT = 80

# ----------------------------
# Enhanced Thread-Safe Frame Reader
# ----------------------------
class FrameReader(threading.Thread):
    def __init__(self, tello):
        super().__init__(daemon=True)
        self.tello = tello
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.queue = []
        self.max_queue_size = 2
        
    def start(self):
        self.running = True
        super().start()
        
    def run(self):
        while self.running:
            try:
                frame = self.tello.get_frame_read().frame
                if frame is not None:
                    with self.lock:
                        if len(self.queue) < self.max_queue_size:
                            self.queue.append(frame)
                        else:
                            self.queue.pop(0)
                            self.queue.append(frame)
            except Exception as e:
                print(f"Frame capture error: {str(e)}")
            time.sleep(0.1)
    
    def get_frame(self):
        with self.lock:
            return self.queue[-1] if self.queue else None
    
    def stop(self):
        self.running = False
        self.join()

# ----------------------------
# RRT Implementation
# ----------------------------
class Node:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent

def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def collision_free(grid, start, end):
    x1, y1 = start
    x2, y2 = end
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    x, y = x1, y1
    sx = 1 if x2 > x1 else -1
    sy = 1 if y2 > y1 else -1
    err = dx - dy

    while True:
        if x < 0 or y < 0 or y >= grid.shape[0] or x >= grid.shape[1]:
            return False
        if grid[y, x] == 1:
            return False
        if x == x2 and y == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return True

def rrt_plan(grid, start, goal):
    h, w = grid.shape
    if not (0 <= start[0] < w and 0 <= start[1] < h) or not (0 <= goal[0] < w and 0 <= goal[1] < h):
        return None

    nodes = [Node(*start)]
    
    for _ in range(RRT_MAX_ITER):
        rand = (random.randint(0, w - 1), random.randint(0, h - 1))
        nearest = min(nodes, key=lambda n: distance((n.x, n.y), rand))
        
        angle = math.atan2(rand[1] - nearest.y, rand[0] - nearest.x)
        new_x = int(nearest.x + RRT_STEP * math.cos(angle))
        new_y = int(nearest.y + RRT_STEP * math.sin(angle))
        
        new_x = max(0, min(w - 1, new_x))
        new_y = max(0, min(h - 1, new_y))
        
        if grid[new_y, new_x] == 0 and collision_free(grid, (nearest.x, nearest.y), (new_x, new_y)):
            new_node = Node(new_x, new_y, nodes.index(nearest))
            nodes.append(new_node)
            
            if distance((new_x, new_y), goal) <= GOAL_RADIUS:
                path = []
                current = new_node
                while current:
                    path.append((current.x, current.y))
                    current = nodes[current.parent] if current.parent is not None else None
                return path[::-1]
    return None

# ----------------------------
# Main Application
# ----------------------------
class DroneController:
    def __init__(self):
        # Initialize Tello
        self.tello = Tello()
        self.tello.connect()
        
        # Initialize video stream
        self.tello.streamon()
        self.frame_reader = FrameReader(self.tello)
        self.frame_reader.start()
        
        # Load YOLO model
        self.model = YOLO(r'D:\EE_Final_Project\EE_Final_Project\yolo-weights\yolov8s-seg.pt', task='detect')
        self.obstacle_classes = [name for id, name in self.model.names.items() if name != TARGET_CLASS]
        
        # Initialize 3D plot
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim(0, FRAME_SIZE[1])
        self.ax.set_ylim(0, FRAME_SIZE[0])
        self.ax.set_zlim(-1, 1)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title("Live 3D Path Planning")
        
        # Initialize plot elements
        self.obstacle_scatter = self.ax.scatter([], [], [], c='black', s=1, alpha=0.3, label='Obstacles')
        self.path_line, = self.ax.plot([], [], [], 'b-', linewidth=2, label='Path')
        self.start_scatter = self.ax.scatter([], [], [], c='green', s=50, label='Start')
        self.goal_scatter = self.ax.scatter([], [], [], c='red', s=50, label='Goal')
        self.ax.legend()
        plt.show(block=False)

        # Takeoff and stabilize
        self.tello.takeoff()
        self.tello.send_rc_control(0, 0, 50, 0)
        time.sleep(2)
        self.tello.send_rc_control(0, 0, 0, 0)

    def update_3d_plot(self, obstacles_img, path, start, goal):
        # Update obstacles
        obs_indices = np.where(obstacles_img == 1)
        obs_x = obs_indices[1]
        obs_y = obs_indices[0]
        obs_z = np.zeros_like(obs_x)
        
        # Update scatter plot data
        self.obstacle_scatter._offsets3d = (obs_x, obs_y, obs_z)
        
        # Update path line
        if path:
            path_3d = [(p[0], p[1], 0) for p in path]
            xs, ys, zs = zip(*path_3d)
        else:
            xs, ys, zs = [], [], []
        self.path_line.set_data(xs, ys)
        self.path_line.set_3d_properties(zs)
        
        # Update start marker
        self.start_scatter._offsets3d = ([start[0]], [start[1]], [0])
        
        # Update goal marker
        if goal is not None:
            self.goal_scatter._offsets3d = ([goal[0]], [goal[1]], [0])
        else:
            self.goal_scatter._offsets3d = ([], [], [])
        
        # Redraw plot
        self.fig.canvas.draw_idle()
        plt.pause(0.1)

    def process_frame(self, frame):
        """Process frame in main thread with GIL"""
        if frame is None:
            return None, None, None, None, None

        # Create thread-safe copy
        frame = frame.copy()
        frame = cv2.resize(frame, (FRAME_SIZE[1], FRAME_SIZE[0]))
        h, w = frame.shape[:2]
        drone_pos = (w // 2, h - 1)

        # Initialize obstacle mask and targets
        obstacle_mask = np.zeros((h, w), dtype=np.uint8)
        targets = []
        
        # Process segmentation results
        results = self.model.predict(frame, conf=0.6)
        
        for i in range(len(results[0])):
            box = results[0].boxes[i]
            mask = results[0].masks[i]
            label = self.model.names[int(box.cls[0])]
            
            if label == TARGET_CLASS:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                tx, ty = (x1 + x2) // 2, (y1 + y2) // 2
                targets.append((tx, ty))
            elif label in self.obstacle_classes:
                polygon = mask.xy[0].astype(np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(frame, [polygon], (0, 0, 0))
                cv2.fillPoly(obstacle_mask, [polygon], 1)

        # Add obstacle margin
        kernel = np.ones((OBSTACLE_MARGIN*2+1, OBSTACLE_MARGIN*2+1), dtype=np.uint8)
        grid = cv2.dilate(obstacle_mask, kernel, iterations=1)

        # Path planning
        path = []
        if targets:
            target = min(targets, key=lambda t: distance(drone_pos, t))
            target = (max(0, min(w-1, target[0])), max(0, min(h-1, target[1])))
            path = rrt_plan(grid, drone_pos, target)

        return frame, grid, path, drone_pos, target if targets else None

    def run(self):
        try:
            while True:
                # Get frame through safe reader
                frame = self.frame_reader.get_frame()
                
                if frame is not None:
                    # Process frame in main thread
                    processed_frame, grid, path, drone_pos, target = self.process_frame(frame)
                    
                    # Update 3D visualization
                    self.update_3d_plot(grid, path, drone_pos, target)

                    # Control logic
                    lr, fb, ud, yv = 0, 0, 0, 0
                    if path and len(path) > 1:
                        current_pos = np.array([drone_pos[0], drone_pos[1]], dtype=np.float64)
                        target_pos = np.array([path[1][0], path[1][1]], dtype=np.float64)
                        
                        # Calculate direction vector
                        direction = target_pos - current_pos
                        direction[1] *= -1  # Flip Y-axis
                        
                        if np.linalg.norm(direction) < MIN_MOVE and len(path) > 2:
                            target_pos = np.array([path[2][0], path[2][1]], dtype=np.float64)
                            direction = target_pos - current_pos
                            direction[1] *= -1

                        norm = np.linalg.norm(direction)
                        if norm > 0:
                            direction = direction / norm
                            fb = int(FORWARD_GAIN * direction[1])
                            lr = int(LATERAL_GAIN * direction[0])

                            # Apply limits
                            fb = max(-100, min(100, fb))
                            lr = max(-100, min(100, lr))

                            # Deadzone handling
                            if abs(fb) < MIN_MOVE: fb = 0
                            if abs(lr) < MIN_MOVE: lr = 0

                    # Height maintenance
                    current_height = self.tello.get_distance_tof()
                    if current_height < SAFE_HEIGHT - 20:
                        ud = 30
                    elif current_height > SAFE_HEIGHT + 20:
                        ud = -30

                    # Send movement commands
                    self.tello.send_rc_control(lr, fb, ud, yv)

                    # Visualization
                    try:
                        if path:
                            for i in range(len(path)-1):
                                cv2.line(processed_frame, path[i], path[i+1], (255,0,0), 2)
                            cv2.circle(processed_frame, path[-1], 8, (0,255,0), -1)
                        cv2.circle(processed_frame, drone_pos, 6, (0,0,255), -1)
                        cv2.imshow("Drone View", processed_frame)
                    except Exception as e:
                        print(f"Visualization error: {str(e)}")

                # Check for exit or emergency stop
                key = cv2.waitKey(1)
                if key == 27:  # ESC
                    break
                elif key == 32:  # SPACE
                    self.tello.emergency()
                    break

        finally:
            # Cleanup
            cv2.destroyAllWindows()
            plt.close()
            try:
                self.frame_reader.stop()
                self.tello.streamoff()
                self.tello.end()
            except Exception as e:
                print(f"Cleanup error: {str(e)}")
            time.sleep(1)
            self.tello.land()

if __name__ == "__main__":
    controller = DroneController()
    controller.run()