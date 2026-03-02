# robot_controller.py
from controller import Robot, Camera
import numpy as np
import cv2
import os

# Initialize robot and timestep
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Initialize camera
camera = robot.getDevice("camera")
camera.enable(timestep)

# Create folder to save images
dataset_dir = "../dataset/images/"
os.makedirs(dataset_dir, exist_ok=True)

frame_id = 0

while robot.step(timestep) != -1:
    # Capture image from camera
    width = camera.getWidth()
    height = camera.getHeight()
    img = np.frombuffer(camera.getImage(), dtype=np.uint8).reshape((height, width, 4))
    
    # Convert BGRA to BGR
    bgr_image = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    # Save image
    cv2.imwrite(os.path.join(dataset_dir, f"frame_{frame_id}.png"), bgr_image)
    frame_id += 1
    
    # Optional: display in OpenCV window
    cv2.imshow("Robot Camera View", bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
