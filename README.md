# Mask R-CNN Custom Dataset Guide (emyeeeel/MaskRCNN-Test)

This repository hosts a **custom module** based on the foundational **Matterport Mask R-CNN** implementation. It provides a focused, step-by-step guide and code structure for setting up, training, and running inference using your own custom object detection and instance segmentation dataset.

## 1. Set Up Your Environment

### 1.1 Clone the Repository
To begin, clone the repository to your local machine:

```bash
git clone https://github.com/emyeeeel/MaskRCNN-Test
```

### 1.2 Create conda environment (If you don't have conda, use python environment)
Rename env_name to convenient environment name for this test.
```bash
conda create -n env_name python=3.7
conda activate env_name
```

### 1.3 Install Dependencies
Ensure you have the required dependencies installed. Run the following commands to install the correct versions of TensorFlow, Keras, and other necessary packages:

```bash
pip install "tensorflow==1.15" "keras==2.3.1" "h5py==2.10.0" "protobuf==3.20.*" "numpy==1.19.5" "cython" "scikit-image" "matplotlib" "imgaug" "opencv-python" "IPython" "pyrealsense2"
```

### 1.4 Verify Installations
To confirm that TensorFlow and Keras are installed correctly, run the following command:

```bash
python -c "import tensorflow as tf; import keras; print(tf.__version__, keras.__version__)"
```

The output should be:
```bash
1.15.0 2.3.1
```


## 2. Prepare Your Dataset

### 2.1 Dataset Folder Structure
Place your dataset in the `samples/custom/dataset` directory. Your dataset should be organized as follows:

* **Train Data:** Add your training images and annotations to the `train` folder.
* **Validation Data:** Add your validation images and annotations to the `val` folder.

**Expected Directory Layout:**
```text
samples/
└── custom/
    └── dataset/
        ├── train/   # Place training images & annotations here
        └── val/     # Place validation images & annotations here
```

### 2.2 Annotation Format
The annotations should be in **COCO-style JSON format**. Ensure that each class in your dataset is correctly labeled in the annotations.



### 2.3 Class Names
In the `custom.py` file, specify the class names that correspond to the categories in your dataset’s annotations.

**Example:**
```python
CLASS_NAMES = ["BG", "metal_bowl", "metal_plate", "metal_snack_plate"]
```

### 2.4 Hyperparameters
Open the `custom.py` file and adjust the following hyperparameters based on your needs:

* **Steps per Epoch:** The number of steps to run per epoch. *Recommended value: 100.*
* **Epochs:** The number of times the model will iterate over the dataset. *Recommended value: 50.*

```python
STEPS_PER_EPOCH = 100
EPOCHS = 50
```

## 3. Training the Model
Once you have prepared your dataset and adjusted the configurations, you can begin training your model.

### 3.1 Run Training
To train the Mask R-CNN model, navigate to the directory where `custom.py` is located and run the following command:

```bash
python custom.py train --dataset="C:\path\to\your\dataset" --weights=coco
```

> **Note:** Replace **"C:\path\to\your\dataset"** with the actual path to your dataset.
> The `--weights=coco` flag will load the pre-trained COCO weights. You can use your own pre-trained weights if available.

## 4. Running Inference
After training your model, you can run inference to make predictions on new images. 

### 4.1 Prepare Input Image
Place the input image you want to test in the `images` folder.

### 4.2 Modify test.py
Open the `test.py` file located in `samples/custom/` and modify the following configuration parameters:

* **Root Directory:** Make sure `ROOT_DIR` is set correctly to the root directory of the project.
* **Weights Path:** Point to the most recent or best-performing model weights in the `logs` folder.
* **Image File:** Specify the path to the image you want to infer.
* **Class Names:** Ensure that the class names in `test.py` match those defined in `custom.py`.

### 4.3 Run Inference
Once you've made the necessary modifications, run the inference script:

For single image input:
```bash
python test.py
```

For live camera (Intel Realsense) inferencing:
```bash
python live.py
```

For live capture (Intel Realsense) inferencing:
```bash
python capture.py
```

This will generate the output image with predictions (**bounding boxes**, **masks**, and **class labels**). 


## Acknowledgements & Citations
This custom dataset module is a modification and direct application of the excellent open-source Mask R-CNN implementation by **Matterport, Inc.** and relies on contributions from others in the community.

### Matterport Mask R-CNN (Base Implementation)
For more details on the original architecture, documentation, and licensing, please visit the Matterport repository.

```
@misc{matterport_maskrcnn_2017,
  title={Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow},
  author={Waleed Abdulla},
  year={2017},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/matterport/Mask_RCNN}},
}
```

### Reference Material (Custom Dataset Guide)
This repository's custom module structure is based on the work of **Soumya Yadav**.

```
@misc{Soumya_Maskrcnn_2020,
  title={Mask R-CNN for custom object detection and instance segmentation on Keras and TensorFlow},
  author={Soumya Yadav},
  year={2020},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/soumyaiitkgp/Mask_RCNN/}},
}
```
