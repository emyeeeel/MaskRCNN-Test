import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # Disable GPU for TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"    # Disable TF warnings

import sys
import skimage.io
import numpy as np
import cv2
import pyrealsense2 as rs
import matplotlib.pyplot as plt

# Mask RCNN imports
ROOT_DIR = r"C:\Users\USER\Documents\Amiel's files\Segmentation Test\MaskRCNN-Test"
sys.path.append(ROOT_DIR)

from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize

# Paths
WEIGHTS_PATH = os.path.join(ROOT_DIR, "logs", "custom20251126T1742", "mask_rcnn_custom_004.h5")

if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(f"Weights file not found: {WEIGHTS_PATH}")

# ==== Configuration ====
class InferenceConfig(Config):
    NAME = "custom"
    NUM_CLASSES = 4              # BG + 3 classes
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

pipeline = rs.pipeline()
config_rs = rs.config()

config_rs.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

print("Starting RealSense camera...")
pipeline.start(config_rs)


# ===================================================================
#                    MAIN VIDEO PROCESSING LOOP
# ===================================================================

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())  # BGR image

        # Convert to RGB for Mask RCNN
        rgb_frame = frame[:, :, ::-1]

        # Run detection
        results = model.detect([rgb_frame], verbose=0)
        r = results[0]

        # Draw results on frame
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

        cv2.imshow("RealSense Mask R-CNN", out)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print("Error:", e)

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("RealSense stopped.")


