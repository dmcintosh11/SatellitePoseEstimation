import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import json
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Import the PoseNet model definition
from modelUQ import PoseNet

# Dataset Definition
class SpeedDataset(Dataset):
    def __init__(self, annotation_items, img_dir, transform=None):
        self.annotations = annotation_items
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        # Construct full image path and load image
        img_path = os.path.join(self.img_dir, ann['filename'])
        # Ensure image exists
        if not os.path.exists(img_path):
            print(f"Warning: Image file not found {img_path}. Skipping.")
            raise FileNotFoundError(f"Image file not found: {img_path}")
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # Convert the quaternion and translation values to tensors
        try:
            q = torch.tensor(ann['q_vbs2tango_true'], dtype=torch.float32) #quaternion
            t = torch.tensor(ann['r_Vo2To_vbs_true'], dtype=torch.float32) #translation
        except:
            q = torch.tensor(ann['q_vbs2tango'], dtype=torch.float32) #quaternion
            t = torch.tensor(ann['r_Vo2To_vbs_true'], dtype=torch.float32) #translation
        return image, q, t

# Loss Function (Normalized Geodesic Distance for rotation, relative translation error for translation since satellite distance can vary greatly between samples)
def pose_loss(pred_rot, pred_trans, true_rot, true_trans, beta=1.0):
    # Quaternion loss using geodesic distance for rotation
    # Normalize quaternions to ensure unit length
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

# Training Loop
def train(model, train_dataloader, optimizer, device, beta_loss):
    model.train()
    total_loss = 0.0
    total_rot_loss = 0.0
    total_trans_loss = 0.0
    
    # Add progress bar for training
    pbar = tqdm(train_dataloader, desc='Training', leave=False)
    for images, true_rot, true_trans in pbar:
        images = images.to(device)
        true_rot = true_rot.to(device)
        true_trans = true_trans.to(device)
        
        optimizer.zero_grad()
        pred_rot, pred_trans = model(images)
        rot_loss, trans_loss = pose_loss(pred_rot, pred_trans, true_rot, true_trans, beta=beta_loss)
        
        #Adds weight to translation loss to prevent rotation loss from overpowering it
        loss = rot_loss + beta_loss * trans_loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_rot_loss += rot_loss.item()
        total_trans_loss += trans_loss.item()
        
        # Update progress bar with current loss
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
    return total_loss / len(train_dataloader), total_rot_loss / len(train_dataloader), total_trans_loss / len(train_dataloader)

#Computes loss on validation set during training to monitor progress
def validate(model, val_dataloader, device, beta_loss, num_mc_samples=1):
    # Activate dropout layers for MC sampling
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()

    total_loss = 0.0
    total_rot_loss = 0.0
    total_trans_loss = 0.0

    # Initialize running sums for variance calculation
    sum_rot = torch.zeros(4, device=device)  # q0, q1, q2, q3
    sum_rot_sq = torch.zeros(4, device=device)
    sum_trans = torch.zeros(3, device=device)  # x, y, z
    sum_trans_sq = torch.zeros(3, device=device)
    total_samples = 0

    with torch.no_grad():
        pbar = tqdm(val_dataloader, desc='Validation', leave=False)
        for images, true_rot, true_trans in pbar:
            images = images.to(device)
            true_rot = true_rot.to(device)
            true_trans = true_trans.to(device)
            
            batch_size = images.size(0)
            total_samples += batch_size

            if num_mc_samples > 1:
                # Initialize batch accumulators on GPU
                batch_sum_rot = torch.zeros((batch_size, 4), device=device)
                batch_sum_rot_sq = torch.zeros((batch_size, 4), device=device)
                batch_sum_trans = torch.zeros((batch_size, 3), device=device)
                batch_sum_trans_sq = torch.zeros((batch_size, 3), device=device)

                # Run MC samples
                mc_pbar = tqdm(range(num_mc_samples), desc='MC Samples', leave=False)
                for _ in mc_pbar:
                    pred_rot_sample, pred_trans_sample = model(images)
                    
                    # Accumulate sums and squared sums for variance calculation
                    batch_sum_rot += pred_rot_sample
                    batch_sum_rot_sq += pred_rot_sample ** 2
                    batch_sum_trans += pred_trans_sample
                    batch_sum_trans_sq += pred_trans_sample ** 2

                # Calculate means for this batch
                mean_pred_rot = batch_sum_rot / num_mc_samples
                mean_pred_trans = batch_sum_trans / num_mc_samples

                # Normalize quaternion
                mean_pred_rot = mean_pred_rot / (torch.norm(mean_pred_rot, p=2, dim=1, keepdim=True) + 1e-8)

                # Calculate batch variances
                batch_var_rot = (batch_sum_rot_sq / num_mc_samples) - (mean_pred_rot ** 2)
                batch_var_trans = (batch_sum_trans_sq / num_mc_samples) - (mean_pred_trans ** 2)

                # Accumulate global statistics
                sum_rot += batch_var_rot.sum(dim=0)
                sum_trans += batch_var_trans.sum(dim=0)

            else:
                mean_pred_rot, mean_pred_trans = model(images)
                mean_pred_rot = mean_pred_rot / (torch.norm(mean_pred_rot, p=2, dim=1, keepdim=True) + 1e-8)
            
            rot_loss, trans_loss = pose_loss(mean_pred_rot, mean_pred_trans, true_rot, true_trans, beta=beta_loss)
            loss = rot_loss + beta_loss * trans_loss
            
            total_loss += loss.item()
            total_rot_loss += rot_loss.item()
            total_trans_loss += trans_loss.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Calculate final average variances
    avg_val_rot_variances = (sum_rot / total_samples).cpu()
    avg_val_trans_variances = (sum_trans / total_samples).cpu()

    # Reset dropout layers to eval mode
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.eval()
            
    return (total_loss / len(val_dataloader), 
            total_rot_loss / len(val_dataloader), 
            total_trans_loss / len(val_dataloader),
            avg_val_rot_variances,
            avg_val_trans_variances)
    

# Helper function to save loss data
def save_loss_data(log_dir, epoch_data):
    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, "loss_data.json")
    with open(file_path, 'w') as f:
        json.dump(epoch_data, f, indent=4)
    print(f"Loss data saved to {file_path}")

# Helper function to plot losses during training
def plot_losses(log_dir, num_epochs, epoch_data, model_name="Model"):
    os.makedirs(log_dir, exist_ok=True)
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 8))
    plt.plot(epochs, epoch_data['train_total_losses'], label='Train Total Loss')
    plt.plot(epochs, epoch_data['val_total_losses'], label='Validation Total Loss')
    plt.title(f'{model_name} - Total Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, 'total_loss_plot.png'))
    plt.close()
    print(f"Total loss plot saved to {os.path.join(log_dir, 'total_loss_plot.png')}")

# New function to plot uncertainty metrics
def plot_uncertainty_metrics(log_dir, num_epochs, epoch_data, model_name="Model"):
    os.makedirs(log_dir, exist_ok=True)
    epochs = range(1, num_epochs + 1)

    # Plot Rotation Variances (Quaternion components)
    plt.figure(figsize=(12, 8))
    for i in range(4):
        if f'val_rot_variance_q{i}' in epoch_data:
            plt.plot(epochs, epoch_data[f'val_rot_variance_q{i}'], label=f'Validation Rot Variance q{i}')
    plt.title(f'{model_name} - Quaternion Component Variances vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Average Variance')
    plt.legend()
    plt.grid(True)
    plt.yscale('log') # Variances can be small, log scale might be useful
    plt.savefig(os.path.join(log_dir, 'rotation_variance_plot.png'))
    plt.close()
    print(f"Rotation variance plot saved to {os.path.join(log_dir, 'rotation_variance_plot.png')}")

    # Plot Translation Variances (x, y, z components)
    plt.figure(figsize=(12, 8))
    trans_labels = ['x', 'y', 'z']
    for i in range(3):
        if f'val_trans_variance_{trans_labels[i]}' in epoch_data:
            plt.plot(epochs, epoch_data[f'val_trans_variance_{trans_labels[i]}'], label=f'Validation Trans Variance {trans_labels[i]}')
    plt.title(f'{model_name} - Translation Component Variances vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Average Variance')
    plt.legend()
    plt.grid(True)
    plt.yscale('log') # Log scale might be useful
    plt.savefig(os.path.join(log_dir, 'translation_variance_plot.png'))
    plt.close()
    print(f"Translation variance plot saved to {os.path.join(log_dir, 'translation_variance_plot.png')}")

# New function to evaluate on the final test set
def evaluate_on_final_test_set(best_model_path, test_annotation_file, test_img_dir, device, beta_loss, 
                               model_name="Model", log_dir=None, architecture='efficientnet_v2_s', 
                               freeze_early_backbone_layers=True, best_val_model_train_loss=None,
                               num_mc_samples=1):
    if not best_model_path or not os.path.exists(best_model_path):
        print(f"Error: Best model path not provided or file not found at {best_model_path}. Skipping final evaluation.")
        return

    print(f"\nStarting final evaluation on the true test set for model: {model_name} (Arch: {architecture}, FreezeEarly: {freeze_early_backbone_layers}) using {best_model_path}")
    
    with open(test_annotation_file, 'r') as f:
        test_ann_items = json.load(f)

    final_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    true_test_dataset = SpeedDataset(test_ann_items, test_img_dir, final_test_transform)
    true_test_dataloader = DataLoader(true_test_dataset, batch_size=64, shuffle=False, num_workers=4)

    model = PoseNet(architecture=architecture, pretrained=False, freeze_early_backbone_layers=freeze_early_backbone_layers).to(device)
    try:
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model from {best_model_path} for final test evaluation: {e}")
        return

    # Activate dropout layers for MC sampling
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()

    total_loss, total_rot_loss, total_trans_loss = 0.0, 0.0, 0.0
    
    # Initialize running sums for variance calculation
    sum_rot = torch.zeros(4, device=device)  # q0, q1, q2, q3
    sum_rot_sq = torch.zeros(4, device=device)
    sum_trans = torch.zeros(3, device=device)  # x, y, z
    sum_trans_sq = torch.zeros(3, device=device)
    total_samples = 0

    with torch.no_grad():
        pbar = tqdm(true_test_dataloader, desc='Testing', leave=True)
        for images, true_rot, true_trans in pbar:
            images, true_rot, true_trans = images.to(device), true_rot.to(device), true_trans.to(device)
            
            batch_size = images.size(0)
            total_samples += batch_size

            if num_mc_samples > 1:
                # Initialize batch accumulators on GPU
                batch_sum_rot = torch.zeros((batch_size, 4), device=device)
                batch_sum_rot_sq = torch.zeros((batch_size, 4), device=device)
                batch_sum_trans = torch.zeros((batch_size, 3), device=device)
                batch_sum_trans_sq = torch.zeros((batch_size, 3), device=device)

                # Run MC samples
                mc_pbar = tqdm(range(num_mc_samples), desc='MC Samples', leave=False)
                for _ in mc_pbar:
                    pred_rot_sample, pred_trans_sample = model(images)
                    
                    # Accumulate sums and squared sums for variance calculation
                    batch_sum_rot += pred_rot_sample
                    batch_sum_rot_sq += pred_rot_sample ** 2
                    batch_sum_trans += pred_trans_sample
                    batch_sum_trans_sq += pred_trans_sample ** 2

                # Calculate means for this batch
                mean_pred_rot = batch_sum_rot / num_mc_samples
                mean_pred_trans = batch_sum_trans / num_mc_samples

                # Normalize quaternion
                mean_pred_rot = mean_pred_rot / (torch.norm(mean_pred_rot, p=2, dim=1, keepdim=True) + 1e-8)

                # Calculate batch variances
                batch_var_rot = (batch_sum_rot_sq / num_mc_samples) - (mean_pred_rot ** 2)
                batch_var_trans = (batch_sum_trans_sq / num_mc_samples) - (mean_pred_trans ** 2)

                # Accumulate global statistics
                sum_rot += batch_var_rot.sum(dim=0)
                sum_trans += batch_var_trans.sum(dim=0)

            else:
                mean_pred_rot, mean_pred_trans = model(images)
                mean_pred_rot = mean_pred_rot / (torch.norm(mean_pred_rot, p=2, dim=1, keepdim=True) + 1e-8)
            
            rot_loss, trans_loss = pose_loss(mean_pred_rot, mean_pred_trans, true_rot, true_trans, beta=beta_loss)
            loss = rot_loss + beta_loss * trans_loss
            
            total_loss += loss.item()
            total_rot_loss += rot_loss.item()
            total_trans_loss += trans_loss.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Calculate final average variances
    avg_var_pred_rot = (sum_rot / total_samples).cpu()
    avg_var_pred_trans = (sum_trans / total_samples).cpu()
    
    # Set model back to eval mode
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.eval()

    avg_total_loss = total_loss / len(true_test_dataloader)
    avg_rot_loss = total_rot_loss / len(true_test_dataloader)
    avg_trans_loss = total_trans_loss / len(true_test_dataloader)

    print(f"--- Final Test Set Performance for {model_name} (MC Samples: {num_mc_samples}) ---")
    print(f"Test Total Loss: {avg_total_loss:.4f}")
    print(f"Test Rotation Loss: {avg_rot_loss:.4f}")
    print(f"Test Translation Loss (Unscaled): {avg_trans_loss:.4f}")
    print(f"Test Translation Loss (Scaled by beta={beta_loss}): {beta_loss * avg_trans_loss:.4f}")
    if best_val_model_train_loss is not None:
        print(f"Training Loss for Best Validation Model Epoch: {best_val_model_train_loss:.4f}")
    
    print(f"Average Quaternion Component Variances: "
          f"q0: {avg_var_pred_rot[0].item():.4e}, q1: {avg_var_pred_rot[1].item():.4e}, "
          f"q2: {avg_var_pred_rot[2].item():.4e}, q3: {avg_var_pred_rot[3].item():.4e}")
    print(f"Average Translation Component Variances: "
          f"x: {avg_var_pred_trans[0].item():.4e}, y: {avg_var_pred_trans[1].item():.4e}, "
          f"z: {avg_var_pred_trans[2].item():.4e}")
    print("---------------------------------------------------\n")

    if log_dir:
        performance_data = {
            "model_name": model_name,
            "test_total_loss": avg_total_loss,
            "test_rotation_loss": avg_rot_loss,
            "test_translation_loss_unscaled": avg_trans_loss,
            "test_translation_loss_scaled": beta_loss * avg_trans_loss,
            "beta_loss_factor": beta_loss,
            "architecture": architecture,
            "freeze_early_backbone_layers": freeze_early_backbone_layers,
            "best_val_model_train_loss": best_val_model_train_loss,
            "num_mc_samples_eval": num_mc_samples,
            "avg_variance_rot_q0": avg_var_pred_rot[0].item() if num_mc_samples > 1 else 0,
            "avg_variance_rot_q1": avg_var_pred_rot[1].item() if num_mc_samples > 1 else 0,
            "avg_variance_rot_q2": avg_var_pred_rot[2].item() if num_mc_samples > 1 else 0,
            "avg_variance_rot_q3": avg_var_pred_rot[3].item() if num_mc_samples > 1 else 0,
            "avg_variance_trans_x": avg_var_pred_trans[0].item() if num_mc_samples > 1 else 0,
            "avg_variance_trans_y": avg_var_pred_trans[1].item() if num_mc_samples > 1 else 0,
            "avg_variance_trans_z": avg_var_pred_trans[2].item() if num_mc_samples > 1 else 0
        }
        performance_file_path = os.path.join(log_dir, "final_test_set_performance.json")
        try:
            with open(performance_file_path, 'w') as f:
                json.dump(performance_data, f, indent=4)
            print(f"Final test set performance saved to {performance_file_path}")
        except Exception as e:
            print(f"Error saving final test set performance: {e}")
    else:
        print("Warning: log_dir not provided. Performance metrics will not be saved to file.")

# Sets up model, loads data, and runs training loop
def main(args):
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define output directories
    model_specific_output_dir = os.path.join(args.output_model_path, args.model_name) 

    log_dir = os.path.join(model_specific_output_dir, "training_logs")
    # Checkpoint dir is no longer used for per-epoch, but best model is saved in model_specific_output_dir
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_specific_output_dir, exist_ok=True)
    
    # Load all initial training annotations
    try:
        with open(args.annotation_file, 'r') as f:
            all_initial_train_annotations = json.load(f)
    except FileNotFoundError:
        print(f"Error: Training annotation file not found at {args.annotation_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from training annotation file: {args.annotation_file}")
        return

    # Split initial training annotations into training and validation sets
    if not (0 < args.val_split_ratio < 1):
        print("Error: --val-split-ratio must be between 0 and 1 (exclusive).")
        return
    
    train_annos, val_annos = train_test_split(
        all_initial_train_annotations, 
        test_size=args.val_split_ratio, 
        random_state=42
    )
    print(f"Split {len(all_initial_train_annotations)} initial train samples into {len(train_annos)} training and {len(val_annos)} validation samples.")

    # Augmentations for the training set
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.0),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # No augmentations for the validation set, only resize, tensor conversion, and normalization
    # This is the same as the final test set transform
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = SpeedDataset(train_annos, args.img_dir, train_transform)
    val_dataset = SpeedDataset(val_annos, args.img_dir, val_transform)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Initialize model based on selected architecture
    model = PoseNet(
        architecture=args.architecture, 
        pretrained=True,
        freeze_early_backbone_layers=args.freeze_early_backbone_layers
    ).to(device)
    
    params_to_optimize = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(params_to_optimize, lr=args.lr)
    
    epoch_data = {
        'train_total_losses': [], 'val_total_losses': [],
        'train_rot_losses': [], 'val_rot_losses': [],
        'train_trans_losses': [], 'val_trans_losses': [],
        # Add keys for validation uncertainty metrics
        'val_rot_variance_q0': [], 'val_rot_variance_q1': [],
        'val_rot_variance_q2': [], 'val_rot_variance_q3': [],
        'val_trans_variance_x': [], 'val_trans_variance_y': [], 'val_trans_variance_z': []
    }
    best_val_loss = float('inf')
    best_val_model_train_loss = None
    best_model_save_path = None

    print(f"Starting training for model: {args.model_name}")
    for epoch in range(args.num_epochs):
        print(f"Starting Epoch {epoch+1}/{args.num_epochs}...")
        train_loss, train_rot_loss, train_trans_loss = train(model, train_dataloader, optimizer, device, args.beta_loss)
        print("\tTraining step done... running validation step...")
        val_loss, val_rot_loss, val_trans_loss, val_rot_variances, val_trans_variances = validate(
            model, val_dataloader, device, args.beta_loss, args.num_mc_samples
        )
        print("\tValidation step done...")
        
        epoch_data['train_total_losses'].append(train_loss)
        epoch_data['train_rot_losses'].append(train_rot_loss)
        epoch_data['train_trans_losses'].append(train_trans_loss)
        epoch_data['val_total_losses'].append(val_loss)
        epoch_data['val_rot_losses'].append(val_rot_loss)
        epoch_data['val_trans_losses'].append(val_trans_loss)

        # Store validation uncertainty metrics
        if args.num_mc_samples > 1:
            for i in range(4):
                epoch_data[f'val_rot_variance_q{i}'].append(val_rot_variances[i].item())
            trans_labels = ['x', 'y', 'z']
            for i in range(3):
                epoch_data[f'val_trans_variance_{trans_labels[i]}'].append(val_trans_variances[i].item())
        else: # Append default 0 if not doing MC sampling
            for i in range(4):
                 epoch_data[f'val_rot_variance_q{i}'].append(0)
            trans_labels = ['x', 'y', 'z']
            for i in range(3):
                epoch_data[f'val_trans_variance_{trans_labels[i]}'].append(0)

        print(f"\nEpoch {epoch+1}/{args.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_model_train_loss = train_loss
            if best_model_save_path:
                try: os.remove(best_model_save_path) 
                except OSError: pass
            
            best_model_save_path = os.path.join(model_specific_output_dir, f"{args.model_name}_best_val_model.pth")
            torch.save(model.state_dict(), best_model_save_path)
            print(f"New best validation model saved to {best_model_save_path} (Val Loss: {best_val_loss:.4f})")
    
    # Save loss data and plot losses
    save_loss_data(log_dir, epoch_data)
    plot_losses(log_dir, args.num_epochs, epoch_data, args.model_name)
    if args.num_mc_samples > 1: # Plot uncertainty only if MC sampling was done
        plot_uncertainty_metrics(log_dir, args.num_epochs, epoch_data, args.model_name)

    # Perform final evaluation on the true test set
    if args.test_annotation_file and args.test_img_dir:
        if os.path.exists(args.test_annotation_file) and os.path.exists(args.test_img_dir):
            if best_model_save_path and os.path.exists(best_model_save_path):
                evaluate_on_final_test_set(
                    best_model_save_path, 
                    args.test_annotation_file, 
                    args.test_img_dir, 
                    device, 
                    args.beta_loss,
                    args.model_name,
                    log_dir,
                    args.architecture,
                    args.freeze_early_backbone_layers,
                    best_val_model_train_loss,
                    args.num_mc_samples
                )
            else:
                print("Warning: Best model was not saved or found. Skipping final test set evaluation.")
        else:
            print("Warning: Final test annotation file or image directory not found. Skipping final test set evaluation.")
    else:
        print("Final test set annotation file or image directory not provided. Skipping final test set evaluation.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PoseNet model with Train/Validation/Test splits.')

    # Data arguments
    parser.add_argument('--annotation-file', type=str, default='../speed/speed/train.json')
    parser.add_argument('--img-dir', type=str, default='../speed/speed/images/train/')
    parser.add_argument('--val-split-ratio', type=float, default=0.2)
    
    parser.add_argument('--test-annotation-file', type=str, default='../speed/speed/lightbox/test.json')
    parser.add_argument('--test-img-dir', type=str, default='../speed/speed/lightbox/images/')

    # Model output argument
    parser.add_argument('--output-model-path', type=str, default='../trained_models')
    parser.add_argument('--model-name', type=str, default='PoseNet_EfficientNet')

    # Training hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta-loss', type=float, default=50.0)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--architecture', type=str, default='efficientnet_v2_s', choices=['efficientnet_v2_s', 'resnet34'])
    parser.add_argument('--freeze-early-backbone-layers', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-mc-samples', type=int, default=50)

    args = parser.parse_args()
    main(args)
