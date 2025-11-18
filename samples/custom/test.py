import os
import sys
import skimage.io
import matplotlib.pyplot as plt

# Mask RCNN imports
ROOT_DIR = r"C:\Users\HP\OneDrive\Documents\TEEP\test-scripts\mask rcnn\Custom_MaskRCNN" #Path to root directory
sys.path.append(ROOT_DIR)

from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize

# Paths
WEIGHTS_PATH = os.path.join(ROOT_DIR, "logs", "custom", "mask_rcnn_custom_0001.h5")
IMAGE_PATH = os.path.join(ROOT_DIR, "images", "bs.png")

# Verify paths
if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(f"Weights file not found: {WEIGHTS_PATH}")
if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Image file not found: {IMAGE_PATH}")

# Configuration
class InferenceConfig(Config):
    NAME = "custom"
    NUM_CLASSES = 4  # Background + 3 classes
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()

# Create model in inference mode
model = modellib.MaskRCNN(mode="inference", config=config, model_dir=os.path.join(ROOT_DIR, "logs"))

# Load weights
print(f"Loading weights from {WEIGHTS_PATH}")
model.load_weights(WEIGHTS_PATH, by_name=True)

# Load image
image = skimage.io.imread(IMAGE_PATH)

# Run detection
results = model.detect([image], verbose=1)
r = results[0]

class_names = ['BG', 'metal_bowl', 'metal_plate', 'metal_snack_plate']

# Visualize results
visualize.display_instances(
    image,
    r['rois'],
    r['masks'],
    r['class_ids'],
    class_names,  
    r['scores']
)

# Optionally, save the output
output_path = os.path.join(ROOT_DIR, "images", "test_image_masked.jpg")
plt.savefig(output_path)
print(f"Masked image saved to {output_path}")
