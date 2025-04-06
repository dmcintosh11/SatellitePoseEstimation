import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import json
import os
import argparse

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
        q = torch.tensor(ann['q_vbs2tango'], dtype=torch.float32) #quaternion
        t = torch.tensor(ann['r_Vo2To_vbs_true'], dtype=torch.float32) #translation
        return image, q, t

# ----------------------------
# Model Definition: PoseNet Variant
# ----------------------------
class PoseNet(nn.Module):
    def __init__(self, pretrained=True, freeze_early_layers=True):
        super(PoseNet, self).__init__()
        # Use ResNet-50 as backbone
        self.backbone = models.resnet50(pretrained=pretrained)
        num_features = self.backbone.fc.in_features
        # Remove the final classification layer
        self.backbone.fc = nn.Identity()

        if freeze_early_layers:
            # Freeze initial layers: conv1, bn1, layer1
            for name, param in self.backbone.named_parameters():
                if name.startswith('conv1') or name.startswith('bn1') or name.startswith('layer1'):
                    param.requires_grad = False
            print("Froze ResNet layers: conv1, bn1, layer1")
        
        # Separate fully connected layers for rotation (quaternion) and translation
        self.fc_rot = nn.Linear(num_features, 4)  # Quaternion output
        self.fc_trans = nn.Linear(num_features, 3)  # Translation output

    def forward(self, x):
        features = self.backbone(x)
        rot = self.fc_rot(features)
        trans = self.fc_trans(features)
        # Normalize quaternion to unit length for a valid rotation
        rot = rot / rot.norm(p=2, dim=1, keepdim=True)
        return rot, trans

# ----------------------------
# Loss Function
# ----------------------------
def pose_loss(pred_rot, pred_trans, true_rot, true_trans, beta=1.0):
    # Quaternion loss using geodesic distance for rotation
    # Normalize quaternions to ensure unit length
    pred_rot_normalized = pred_rot / torch.norm(pred_rot, dim=1, keepdim=True)
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
    
    # Combine losses with beta weighting factor
    return loss_rot + beta * loss_trans

# ----------------------------
# Training Loop
# ----------------------------
def train(model, train_dataloader, optimizer, device, beta_loss):
    model.train()
    total_loss = 0.0
    for images, true_rot, true_trans in train_dataloader:
        images = images.to(device)
        true_rot = true_rot.to(device)
        true_trans = true_trans.to(device)
        
        optimizer.zero_grad()
        pred_rot, pred_trans = model(images)
        loss = pose_loss(pred_rot, pred_trans, true_rot, true_trans, beta=beta_loss)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(train_dataloader)

def test(model, test_dataloader, device, beta_loss):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, true_rot, true_trans in test_dataloader:
            images = images.to(device)
            true_rot = true_rot.to(device)
            true_trans = true_trans.to(device)
            
            pred_rot, pred_trans = model(images)
            loss = pose_loss(pred_rot, pred_trans, true_rot, true_trans, beta=beta_loss)
            total_loss += loss.item()
    return total_loss / len(test_dataloader)
    
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
    
    model = PoseNet(pretrained=True, freeze_early_layers=True).to(device)
    
    # Filter parameters to only include those that require gradients
    params_to_optimize = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(params_to_optimize, lr=args.lr)
    
    # Use num_epochs from args
    for epoch in range(args.num_epochs):
        print(f"Starting Epoch {epoch+1}/{args.num_epochs}...")
        train_loss = train(model, train_dataloader, optimizer, device, args.beta_loss)
        test_loss = test(model, test_dataloader, device, args.beta_loss)
        print(f"Epoch {epoch+1}/{args.num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    
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
    parser.add_argument('--beta-loss', type=float, default=1.0, help='Weighting factor for translation loss.')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training and testing.')
    parser.add_argument('--num-epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for DataLoader.')

    # Device argument
    parser.add_argument('--force-cpu', action='store_true', help='Force using CPU even if CUDA is available.')

    args = parser.parse_args()
    main(args)
