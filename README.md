# Satellite Pose Estimation from Synthetic Imagery

## Project Overview

This project focuses on estimating the 6-DoF pose (position and orientation) of a satellite from synthetic imagery using deep learning. It utilizes the [SPEED (Satellite Pose Estimation Dataset)](https://arxiv.org/abs/1807.01136) dataset, which contains rendered images of a satellite with corresponding ground truth pose information. The core of the project is a `PoseNet`-based model implemented in PyTorch, adapted for regression of quaternion (orientation) and translation vectors (position).

This project serves as a demonstration of applying computer vision and deep learning techniques to space-based applications, relevant for roles involving machine learning in the space economy.

## Key Features

*   **Pose Estimation Model:** Implements a ResNet-50 backbone based `PoseNet` architecture for regressing 6-DoF pose.
*   **Dataset Handling:** Includes a PyTorch `Dataset` and `DataLoader` for the SPEED dataset.
*   **Training Script:** Provides a script (`train.py`) to train the pose estimation model.
*   **Customizable Training:** Allows configuration of hyperparameters like learning rate, batch size, epochs, and loss weighting via command-line arguments.
*   **Deployment Strategy:** Designed for deployment on cloud compute (e.g., AWS) via Git and Docker.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd SatellitePoseEstimation
    ```
2.  **Set up a Python environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install torch torchvision Pillow # Add other dependencies if needed, e.g., numpy
    ```
    *(Consider creating a `requirements.txt` file for easier dependency management)*

4.  **Download the SPEED dataset:** Obtain the dataset and organize it as expected by the `train.py` script (image folders and annotation JSON files).

## Usage

To train the model, run the `train.py` script with the appropriate arguments pointing to your dataset locations and desired hyperparameters.

**Example:**

```bash
python train.py \
    --annotation-file /path/to/your/train_annotations.json \
    --img-dir /path/to/your/train_images/ \
    --test-annotation-file /path/to/your/test_annotations.json \
    --test-img-dir /path/to/your/test_images/ \
    --output-model-path trained_models/posenet_speed_v1.pth \
    --lr 1e-4 \
    --beta-loss 100.0 \
    --batch-size 64 \
    --num-epochs 25 \
    --num-workers 8
```

**Command-line Arguments:**

*   `--annotation-file`: Path to the training annotation JSON file.
*   `--img-dir`: Path to the training image directory.
*   `--test-annotation-file`: Path to the test annotation JSON file.
*   `--test-img-dir`: Path to the test image directory.
*   `--output-model-path`: Path to save the trained model weights (default: `posenet_speed.pth`).
*   `--lr`: Learning rate (default: `1e-4`).
*   `--beta-loss`: Weighting factor for the translation loss component (default: `1.0`). Adjust this based on the relative scale of rotation and translation errors.
*   `--batch-size`: Batch size for training and testing (default: `32`).
*   `--num-epochs`: Number of training epochs (default: `10`).
*   `--num-workers`: Number of workers for data loading (default: `4`).
*   `--force-cpu`: Force using CPU even if CUDA is available.

## Deployment

The current deployment strategy involves:

1.  Pushing the code to a Git repository.
2.  Pulling the repository onto an AWS EC2 instance (or similar cloud compute).
3.  Building a Docker container that includes the code, dependencies, and necessary environment setup.
4.  Running the training or inference process within the Docker container on the cloud instance.

*(Further MLOps integration could involve tools like AWS SageMaker, MLflow, or Kubeflow for experiment tracking, model versioning, and automated pipelines.)*

## Future Work & Potential Improvements

*   **Uncertainty Quantification:** Implement methods to estimate the model's confidence in its pose predictions.
*   **Advanced Architectures:** Explore more recent or specialized architectures for pose estimation.
*   **Data Augmentation:** Apply more sophisticated data augmentation techniques relevant to satellite imagery (e.g., lighting variations, occlusion).
*   **MLOps Integration:** Build a more robust MLOps pipeline for automated training, evaluation, and deployment.
*   **Edge Deployment:** Optimize the model (e.g., quantization, pruning) for deployment on edge devices.
*   **Requirements File:** Add a `requirements.txt` for explicit dependency management.
*   **Dockerization:** Provide a `Dockerfile` for easier environment setup and deployment.
*   **Evaluation Metrics:** Implement more detailed pose estimation evaluation metrics (e.g., mean translation error, mean rotation error).

## License

*(Specify your chosen license here, e.g., MIT License)*