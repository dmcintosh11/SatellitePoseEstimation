import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import json
import os
import argparse
# from torchvision.models import ResNet50_Weights # This is now handled within model.py

# Import the PoseNet model definition
from model import PoseNet

# ----------------------------
# Dataset Definition
# ----------------------------
class SpeedDataset(Dataset):
    def __init__(self, annotation_file, img_dir, transform=None):
        with open(annotation_file, 'r') as f:
            # Assumes a list of dictionaries in the JSON file
            self.annotations = json.load(f)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        # Construct full image path and load image
        img_path = os.path.join(self.img_dir, ann['filename'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # Convert the quaternion and translation values to tensors
        # Ensure your JSON stores these as lists of numbers.
        q = torch.tensor(ann['q_vbs2tango_true'], dtype=torch.float32) #quaternion
        t = torch.tensor(ann['r_Vo2To_vbs_true'], dtype=torch.float32) #translation
        return image, q, t

# # ----------------------------
# # Model Definition: PoseNet Variant
# # ----------------------------
# class PoseNet(nn.Module):
#     def __init__(self, pretrained=True, freeze_early_layers=True, dropout_rate=0.3):
#         super(PoseNet, self).__init__()
#         # Use ResNet-50 as backbone
#         self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
#         num_features = self.backbone.fc.in_features
#         # Remove the final classification layer
#         self.backbone.fc = nn.Identity()

#         if freeze_early_layers:
#             # Freeze initial layers: conv1, bn1, layer1
#             for name, param in self.backbone.named_parameters():
#                 if name.startswith('conv1') or name.startswith('bn1') or name.startswith('layer1'):
#                     param.requires_grad = False
#             print("Froze ResNet layers: conv1, bn1, layer1")
        
#         # Shared feature processing layers
#         self.shared_layers = nn.Sequential(
#             nn.Linear(num_features, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(1024, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate)
#         )
        
#         # Rotation-specific layers
#         self.rot_layers = nn.Sequential(
#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(256, 128),
#             nn.ReLU()
#         )
#         self.fc_rot = nn.Linear(128, 4)  # Quaternion output
        
#         # Translation-specific layers
#         self.trans_layers = nn.Sequential(
#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(256, 128),
#             nn.ReLU()
#         )
#         self.fc_trans = nn.Linear(128, 3)  # Translation output

#     def forward(self, x):
#         features = self.backbone(x)
#         shared_features = self.shared_layers(features)
        
#         # Rotation branch
#         rot_features = self.rot_layers(shared_features)
#         rot = self.fc_rot(rot_features)
#         # Normalize quaternion to unit length for a valid rotation
#         rot = rot / rot.norm(p=2, dim=1, keepdim=True)
        
#         # Translation branch
#         trans_features = self.trans_layers(shared_features)
#         trans = self.fc_trans(trans_features)
        
#         return rot, trans
    
# class PoseNet(nn.Module):
#     # NOTE: We set pretrained=False and freeze_early_layers=False here
#     # because we are loading *already trained* weights, not initializing
#     # from scratch or ImageNet. The state dict contains the trained weights
#     # for all layers as they were during training.
#     def __init__(self, pretrained=False, freeze_early_layers=False):
#         super(PoseNet, self).__init__()
#         # Load architecture only. Use weights=None for newer torchvision
#         # or pretrained=False for older versions. Adapt if needed.
#         try:
#             self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
#         except TypeError:
#             self.backbone = models.resnet50(weights=None) # Fallback for older torchvision
#             
#         num_features = self.backbone.fc.in_features
#         self.backbone.fc = nn.Identity()

#         # Recreate the same layers as during training
#         self.fc_rot = nn.Linear(num_features, 4)  # Quaternion output
#         self.fc_trans = nn.Linear(num_features, 3)  # Translation output

#     def forward(self, x):
#         features = self.backbone(x)
#         rot = self.fc_rot(features)
#         trans = self.fc_trans(features)
#         # Normalize quaternion to unit length for a valid rotation
#         # Add a small epsilon to prevent division by zero if norm is zero
#         norm = rot.norm(p=2, dim=1, keepdim=True)
#         rot = rot / (norm + 1e-8)
#         return rot, trans

# ----------------------------
# Loss Function
# ----------------------------
def pose_loss(pred_rot, pred_trans, true_rot, true_trans, beta=1.0):
    # Quaternion loss using geodesic distance for rotation
    # Normalize quaternions to ensure unit length
    # pred_rot_normalized = pred_rot / torch.norm(pred_rot, dim=1, keepdim=True) # Already normalized in forward pass
    pred_rot_normalized = pred_rot
    true_rot_normalized = true_rot / torch.norm(true_rot, dim=1, keepdim=True)
    
    # Calculate dot product between predicted and true quaternions
    dot_product = torch.sum(pred_rot_normalized * true_rot_normalized, dim=1)
    
    # Clamp dot product to valid range for arccos
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    
    # Geodesic distance (angular distance in radians)
    # We use 2 * arccos(|q1 · q2|) as the distance between quaternions
    # The absolute value handles the fact that q and -q represent the same rotation
    angular_distance = 2.0 * torch.acos(torch.abs(dot_product))
    loss_rot = torch.mean(angular_distance ** 2)  # Squared angular distance
    
    # Calculate relative translation error (as percentage of distance)
    # This is more appropriate for satellite pose estimation where distances can vary
    true_trans_norm = torch.norm(true_trans, dim=1, keepdim=True)
    # Add small epsilon to avoid division by zero
    epsilon = 1e-6
    relative_trans_error = torch.norm(pred_trans - true_trans, dim=1) / (true_trans_norm.squeeze() + epsilon)
    loss_trans = torch.mean(relative_trans_error ** 2)  # Squared relative error
    
    return loss_rot, loss_trans

# ----------------------------
# Training Loop
# ----------------------------
def train(model, train_dataloader, optimizer, device, beta_loss):
    model.train()
    total_loss = 0.0
    total_rot_loss = 0.0
    total_trans_loss = 0.0
    for images, true_rot, true_trans in train_dataloader:
        images = images.to(device)
        true_rot = true_rot.to(device)
        true_trans = true_trans.to(device)
        
        optimizer.zero_grad()
        pred_rot, pred_trans = model(images)
        rot_loss, trans_loss = pose_loss(pred_rot, pred_trans, true_rot, true_trans, beta=beta_loss)
        loss = rot_loss + beta_loss * trans_loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_rot_loss += rot_loss.item()
        total_trans_loss += trans_loss.item()
        
    return total_loss / len(train_dataloader), total_rot_loss / len(train_dataloader), total_trans_loss / len(train_dataloader)

def test(model, test_dataloader, device, beta_loss):
    model.eval()
    total_loss = 0.0
    total_rot_loss = 0.0
    total_trans_loss = 0.0
    with torch.no_grad():
        for images, true_rot, true_trans in test_dataloader:
            images = images.to(device)
            true_rot = true_rot.to(device)
            true_trans = true_trans.to(device)
            
            pred_rot, pred_trans = model(images)
            rot_loss, trans_loss = pose_loss(pred_rot, pred_trans, true_rot, true_trans, beta=beta_loss)
            loss = rot_loss + beta_loss * trans_loss
            
            total_loss += loss.item()
            total_rot_loss += rot_loss.item()
            total_trans_loss += trans_loss.item()
            
    return total_loss / len(test_dataloader), total_rot_loss / len(test_dataloader), total_trans_loss / len(test_dataloader)
    
# ----------------------------
# Main Function to Set Up and Run Training
# ----------------------------
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Define image transformations
    # Augmentations for the training set
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)), # Resize directly to final size
        # Removed RandomCrop and RandomRotation as they invalidate pose labels without adjustment
        transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.0), # Reduced color jitter for space imaging
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)), # Add Gaussian blur augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # No augmentations for the test set, only resize, tensor conversion, and normalization
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Use paths from args and apply respective transforms
    train_dataset = SpeedDataset(args.annotation_file, args.img_dir, train_transform)
    test_dataset = SpeedDataset(args.test_annotation_file, args.test_img_dir, test_transform)
    
    # Use dataloader args
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    model = PoseNet(pretrained=True, freeze_early_layers=False).to(device)
    
    # Filter parameters (optional now, but harmless)
    params_to_optimize = filter(lambda p: p.requires_grad, model.parameters())
    # Ensure learning rate from args is passed to optimizer
    optimizer = optim.Adam(params_to_optimize, lr=args.lr)
    
    # Use num_epochs from args
    for epoch in range(args.num_epochs):
        print(f"Starting Epoch {epoch+1}/{args.num_epochs}...")
        train_loss, train_rot_loss, train_trans_loss = train(model, train_dataloader, optimizer, device, args.beta_loss)
        test_loss, test_rot_loss, test_trans_loss = test(model, test_dataloader, device, args.beta_loss)
        print(f"\n\nEpoch {epoch+1}/{args.num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}\nTrain Rot Loss: {train_rot_loss:.4f}, Test Rot Loss: {test_rot_loss:.4f}\nTrain Scaled Trans Loss: {args.beta_loss * train_trans_loss:.4f}, Test Scaled Trans Loss: {args.beta_loss * test_trans_loss:.4f}")
    
    # Save the trained model using output path from args
    print(f"Saving model to {args.output_model_path}")
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_model_path), exist_ok=True)
    torch.save(model.state_dict(), args.output_model_path)
    print("Model saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PoseNet model.')

    # Data arguments
    parser.add_argument('--annotation-file', type=str, required=True, help='Path to the training annotation JSON file.')
    parser.add_argument('--img-dir', type=str, required=True, help='Path to the training image directory.')
    parser.add_argument('--test-annotation-file', type=str, required=True, help='Path to the test annotation JSON file.')
    parser.add_argument('--test-img-dir', type=str, required=True, help='Path to the test image directory.')

    # Model output argument
    parser.add_argument('--output-model-path', type=str, default='posenet_speed.pth', help='Path to save the trained model weights.')

    # Training hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--beta-loss', type=float, default=50.0, help='Weighting factor for translation loss.')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training and testing.')
    parser.add_argument('--num-epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for DataLoader.')

    # Device argument
    parser.add_argument('--force-cpu', action='store_true', help='Force using CPU even if CUDA is available.')

    args = parser.parse_args()
    main(args)
