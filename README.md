# Autonomous Drone Tracking System
This project is focused on developing an algorithm that enables a drone to autonomously track a user along a designated route, while dynamically detecting and avoiding obstacles. The project integrates computer vision, data processing methodologies, and advanced motion planning algorithms to ensure safe and efficient navigation.

# Project Overview
Purpose : The primary goal of this project is to create an autonomous drone tracking system that identifies a user's initial position and destination, and continuously generates an optimal path connecting these points. The algorithm is designed to detect obstacles in real-time and compute the shortest and safest trajectory to the destination, avoiding any obstacles encountered along the way.

# Key Features
1) Autonomous Tracking Algorithm: Developed an algorithm that integrates computer vision, data processing, and motion planning techniques to enable autonomous tracking.

2) Real-time Detection: Utilized the YOLOv8 algorithm for real-time detection of the user and obstacles in the environment.

3) Motion Planning: Implemented the Rapidly Exploring Random Tree (RRT) algorithm to continuously generate an optimal path considering dynamic obstacles.

4) Visual Feedback: Displayed identified obstacles and the computed optimal path on the user's laptop screen for guidance during navigation.

# Technical Details
1) Computer Vision: Frames from the drone's camera are processed using OpenCV to detect the user. The YOLOv8 algorithm is used to identify both static and dynamic obstacles.

2) Feedback Loop: A feedback loop calculates the user's position and error, guiding the drone's tracking behavior.

3) Path Planning: The RRT algorithm processes the user's binary contour and YOLO's output to identify obstacles and compute the optimal path from start to destination.

4) User Interface: The computed RRT graph and detected obstacles are displayed on the laptop screen, providing real-time navigation guidance.

# Project Deliverables
1) Autonomous Tracking Algorithm: Integration of computer vision techniques, data processing, and motion planning algorithms.

2) Real-time Obstacle Detection: Using the YOLOv8 algorithm to detect users and obstacles in the environment.

3) Optimal Path Generation: Utilizing the RRT algorithm to generate an optimal path from the user's current position to the destination.

4) Visual Display: Real-time display of identified obstacles and computed optimal paths on the user's laptop screen.

# Running the Code
Prerequisites:
- **Python**: Ensure Python is installed on your system.
- **Dependencies**: Install necessary libraries using the `requirements.txt` file provided in `python_codes` library.

# Steps to Run the Project
1) **Connect to the Drone**:
    - Ensure your laptop is connected to the drone via Wi-Fi.
2) **Run the Python Script**:
    - Execute the `final_code.py` file to start the drone's autonomous navigation and obstacle avoidance.

# Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

# Contact
For any inquiries, please contact :
[Emil Elasmar](mailto:emil.elasmar1@gmail.com).