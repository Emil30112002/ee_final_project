import numpy as np
import cv2
from ultralytics import YOLO
import time

# Load the YOLO model once
model = YOLO(r"C:\Users\micha\Downloads\autonomous-drone-final-project-master\autonomous-drone-final-project-master\yolo-weights\yolov8s-seg.pt")

def detect_objects(img):
    if img is None:
        print("Input image is None. Exiting function.")
        return 0

    # Start timing
    start_time = time.time()

    height, width, channels = img.shape
    results = model.predict(source=img.copy(), save=False, save_txt=False)

    segmentation_contours_idx = []
    for result in results:
        for seg in result.masks.xyn:
            # Convert normalized coordinates to pixel values
            seg[:, 0] *= width
            seg[:, 1] *= height
            segment = np.array(seg, dtype=np.int32)
            segmentation_contours_idx.append(segment)

    # Draw the segmentation contours on the image
    for seg in segmentation_contours_idx:
        try:
            cv2.fillPoly(img, [seg], (0, 0, 0))  # Fill the detected area with black
        except Exception as e_inner:
            print(f"Error while processing object: {e_inner}")

    # Save the processed image
    drone_frame_path = r"C:\Users\micha\Downloads\autonomous-drone-final-project-master\autonomous-drone-final-project-master\test_pics\pic_after_object_detection.jpg"
    cv2.imwrite(drone_frame_path, img)

    # Print the time taken for detection
    print(f"Detection completed in {time.time() - start_time:.2f} seconds")

# Load and process the image
img = cv2.imread(r"C:\Users\micha\Downloads\autonomous-drone-final-project-master\autonomous-drone-final-project-master\test_pics\pic.jpg")
detect_objects(img)