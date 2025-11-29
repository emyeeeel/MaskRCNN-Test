import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   
import sys
import cv2
import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt

# Mask RCNN imports
ROOT_DIR = r"C:\Users\HP\OneDrive\Documents\TEEP\MaskRCNN\Plate Segmentation\MaskRCNN-Test"  # change this to root file directory
sys.path.append(ROOT_DIR)

from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize

# Paths
WEIGHTS_PATH = os.path.join(ROOT_DIR, "logs", "custom", "mask_rcnn_custom_0004.h5")  # change the weight folder and weight path

if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(f"Weights file not found: {WEIGHTS_PATH}")

# ==== Configuration ====
class InferenceConfig(Config):
    NAME = "custom"
    NUM_CLASSES = 4  # BG + 3 classes
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()

# ==== Load Mask R-CNN Model ====
print(f"Loading weights from {WEIGHTS_PATH}")
model = modellib.MaskRCNN(mode="inference", config=config, model_dir=os.path.join(ROOT_DIR, "logs"))
model.load_weights(WEIGHTS_PATH, by_name=True)

class_names = ['BG', 'metal_bowl', 'metal_plate', 'metal_snack_plate']


# ===================================================================
#                    REALSENSE INPUT SETUP
# ===================================================================

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config_rs = rs.config()

# Enable color stream
config_rs.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

# Start the RealSense pipeline
print("Starting RealSense camera...")
pipeline.start(config_rs)

# ===================================================================
#                    CAPTURE SINGLE FRAME
# ===================================================================

try:
    print("Capturing image from RealSense...")
    # Wait for a valid frame
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    if not color_frame:
        raise ValueError("No color frame captured.")

    # Convert to numpy array (BGR image)
    frame = np.asanyarray(color_frame.get_data())

    # Save the captured frame as an image (optional, for debugging)
    cv2.imwrite('captures/captured_image.jpg', frame)

    # Convert to RGB (Mask R-CNN expects RGB input)
    rgb_frame = frame[:, :, ::-1]

    # Run Mask R-CNN detection
    results = model.detect([rgb_frame], verbose=0)
    r = results[0]

    # Draw results on the image
    masked_image = visualize.apply_mask(
        image=frame.copy(),
        mask=r['masks'][:, :, 0] if r['masks'].shape[-1] > 0 else np.zeros_like(frame[:, :, 0]),
        color=(1, 0, 0),
        alpha=0.3
    ) if r['masks'].shape[-1] > 0 else frame

    # Overlay full instance visualization
    out = visualize.display_instances_cv2(
        masked_image,
        r['rois'],
        r['masks'],
        r['class_ids'],
        class_names,
        r['scores']
    )

    # Show the processed image
    cv2.imshow("Mask R-CNN Output", out)

    # Wait for a key press to close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except Exception as e:
    print("Error:", e)

finally:
    # Stop the RealSense pipeline
    pipeline.stop()
    print("RealSense stopped.")