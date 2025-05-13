import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import os

# Import the PoseNet model definition
from model import PoseNet 

# Use environment variable for model path, default to 'posenet_speed.pth'
MODEL_PATH = os.environ.get('MODEL_PATH', '../models/posenet_speed_run_EN.pth')
# Use environment variable for architecture, default to 'efficientnet_v2_s'
# Ensure this matches the architecture of the loaded MODEL_PATH
MODEL_ARCHITECTURE = os.environ.get('MODEL_ARCHITECTURE', 'efficientnet_v2_s') 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Preprocessing transform for inference
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load the trained model
def load_model(model_path=MODEL_PATH, architecture=MODEL_ARCHITECTURE):
    if not os.path.exists(model_path):
        print(f"Error: Model weights file not found at {model_path}")
        return None
        
    print(f"Loading model from {model_path} for device: {DEVICE} (Arch: {architecture})")
    try:
        model = PoseNet(
            architecture=architecture,
            pretrained=False
        )
        
        # Load the saved state dictionary
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval() 
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Global model variable (load once when script is imported/run for app.py)
loaded_model = load_model()

# Runs image through model
def predict_pose(image_bytes):
    """
    Takes image bytes, preprocesses the image, runs inference,
    and returns the predicted pose (quaternion, translation).
    Returns None, None if model not loaded or prediction fails.
    """
    if loaded_model is None:
        print("Model not loaded. Prediction aborted.")
        return None, None
        
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_tensor = preprocess(image).unsqueeze(0) # Add batch dimension
        image_tensor = image_tensor.to(DEVICE)

        with torch.no_grad(): # Disable gradient calculation for inference
            pred_rot, pred_trans = loaded_model(image_tensor)

        # Move results to CPU and convert to list for JSON serialization
        pred_rot_list = pred_rot.squeeze().cpu().numpy().tolist()
        pred_trans_list = pred_trans.squeeze().cpu().numpy().tolist()

        return pred_rot_list, pred_trans_list

    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None