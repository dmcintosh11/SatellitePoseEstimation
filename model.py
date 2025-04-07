import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights # Make sure this import is present if used

# ----------------------------
# Model Definition: PoseNet Variant
# ----------------------------
# Using the simpler version that seems active in train.py currently.
# If the more complex one (with shared_layers, rot_layers, trans_layers) 
# should be used, replace this definition with that one.
class PoseNet(nn.Module):
    # NOTE: The parameters `pretrained` and `freeze_early_layers` are relevant
    # during TRAINING initialization. When loading a saved model for INFERENCE,
    # you typically initialize with pretrained=False/weights=None as the trained
    # weights are loaded from the state_dict.
    def __init__(self, pretrained=True, freeze_early_layers=True):
        super(PoseNet, self).__init__()
        
        # Determine how to load weights based on torchvision version and `pretrained` flag
        resnet_weights = None
        if pretrained:
             # Use the modern way with specific weights enum if available
            if hasattr(models, 'ResNet50_Weights'):
                 resnet_weights = models.ResNet50_Weights.DEFAULT
            else:
                 # Fallback for older torchvision, directly uses boolean
                 # Note: this way might be deprecated in future torchvision
                 resnet_weights = pretrained # Pass True/False directly

        # Load backbone architecture, potentially with pretrained weights
        # The 'weights' argument is preferred in newer torchvision
        try:
             self.backbone = models.resnet50(weights=resnet_weights)
        except TypeError: # Handle older torchvision that might use 'pretrained=' boolean
             self.backbone = models.resnet50(pretrained=bool(resnet_weights))


        num_features = self.backbone.fc.in_features
        # Remove the final classification layer
        self.backbone.fc = nn.Identity()

        if freeze_early_layers and pretrained: # Only freeze if using pretrained weights
            # Freeze initial layers: conv1, bn1, layer1
            print("Freezing ResNet layers: conv1, bn1, layer1")
            for name, param in self.backbone.named_parameters():
                # Make freezing logic more robust by checking layer names carefully
                layer_name_parts = name.split('.')
                if len(layer_name_parts) > 0:
                    if layer_name_parts[0] in ['conv1', 'bn1'] or \
                       (layer_name_parts[0] == 'layer1'): # Freeze the entire first layer block
                         param.requires_grad = False
        elif freeze_early_layers and not pretrained:
             print("Warning: freeze_early_layers=True but pretrained=False. Not freezing any layers.")

        # Recreate the same layers as during training
        self.fc_rot = nn.Linear(num_features, 4)  # Quaternion output
        self.fc_trans = nn.Linear(num_features, 3)  # Translation output

    def forward(self, x):
        features = self.backbone(x)
        rot = self.fc_rot(features)
        trans = self.fc_trans(features)
        # Normalize quaternion to unit length for a valid rotation
        # Add a small epsilon to prevent division by zero if norm is zero
        norm = rot.norm(p=2, dim=1, keepdim=True)
        rot = rot / (norm + 1e-8) # Add epsilon for numerical stability
        return rot, trans

# # If you intend to use the more complex PoseNet version (currently commented out 
# # in train.py), uncomment it here and comment out the simpler one above.
# # Make sure to adjust the __init__ and forward methods accordingly.
# class PoseNetComplex(nn.Module):
#      def __init__(self, pretrained=True, freeze_early_layers=True, dropout_rate=0.3):
#           # ... (rest of the complex model definition) ...
#          pass
#      def forward(self, x):
#           # ... (forward pass for complex model) ...
#          pass