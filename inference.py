import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import os # Added for model path flexibility

# Import the PoseNet model definition
from model import PoseNet 

# ----------------------------
# Inference Setup
# ----------------------------
# Use environment variable for model path, default to 'posenet_speed.pth'
MODEL_PATH = os.environ.get('MODEL_PATH', '../models/posenet_speed_run_EN.pth')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Preprocessing transform for inference (should match test transform from training)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load the trained model
def load_model(model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        print(f"Error: Model weights file not found at {model_path}")
        print("Please ensure the trained model file is present or set the MODEL_PATH environment variable.")
        return None
        
    print(f"Loading model from {model_path} for device: {DEVICE}")
    try:
        # Initialize model architecture using the imported class.
        # Set pretrained=False as we're loading a state dict, not ImageNet weights here.
        model = PoseNet(pretrained=False, freeze_early_layers=False)
        
        # Load the saved state dictionary
        # Use map_location to ensure compatibility if trained on GPU but inferring on CPU
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval() # Set model to evaluation mode
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Global model variable (load once when script is imported/run)
loaded_model = load_model()

# ----------------------------
# Prediction Function
# ----------------------------
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
        # Potentially log the error in a real application
        return None, None

if __name__ == '__main__':
    # Example usage: Load an image and predict
    # Replace 'path/to/your/test_image.png' with an actual image path
    # Make sure posenet_speed.pth (or path set by MODEL_PATH env var) exists
    example_image_path = 'example_image.png' # Placeholder path
    if not os.path.exists(example_image_path):
         print(f"Example image '{example_image_path}' not found. Skipping example prediction.")
    elif loaded_model is None:
        print("Model could not be loaded. Skipping example prediction.")
    else:
        try:
            with open(example_image_path, 'rb') as f:
                img_bytes = f.read()

            print("\nRunning example prediction...")
            q, t = predict_pose(img_bytes)

            if q and t:
                print(f"Predicted Quaternion (q_vbs2tango): {q}")
                print(f"Predicted Translation (r_Vo2To_vbs): {t}")
            else:
                print("Prediction failed.")

        except Exception as e:
            print(f"An error occurred during example run: {e}")