import torch
import torch.nn as nn
from torchvision import models
# Make sure weight enums are imported if used directly
from torchvision.models import EfficientNet_V2_S_Weights, ResNet50_Weights

# ----------------------------
# Model Definition: PoseNet Variant with EfficientNetV2-S Backbone
# ----------------------------
class PoseNet(nn.Module):
    def __init__(self, pretrained=True, freeze_early_layers=False): # freeze_early_layers kept in signature but not used currently
        super(PoseNet, self).__init__()
        
        # Determine how to load weights for EfficientNetV2-S
        effnet_weights = None
        if pretrained:
            if hasattr(models, 'EfficientNet_V2_S_Weights'):
                 effnet_weights = models.EfficientNet_V2_S_Weights.DEFAULT
            else:
                 # Fallback might just be `pretrained=True` if enum doesn't exist
                 effnet_weights = pretrained # Boolean for older torchvision?

        # Load EfficientNetV2-S backbone
        try:
            effnet_backbone = models.efficientnet_v2_s(weights=effnet_weights)
        except TypeError:
            # Fallback for older torchvision that might expect boolean `pretrained`
            effnet_backbone = models.efficientnet_v2_s(pretrained=bool(effnet_weights))

        # Extract features before the final classifier layer
        # For EfficientNetV2, the classifier is typically `effnet_backbone.classifier`
        # We need the number of input features to the last linear layer
        if isinstance(effnet_backbone.classifier, nn.Sequential) and len(effnet_backbone.classifier) > 0:
            # Access the last layer of the classifier sequence
            final_layer = effnet_backbone.classifier[-1]
            if isinstance(final_layer, nn.Linear):
                num_features = final_layer.in_features
                # Replace the classifier with Identity
                effnet_backbone.classifier = nn.Identity()
            else:
                raise TypeError("EfficientNetV2 classifier's last layer is not Linear.")
        else:
            raise TypeError("EfficientNetV2 classifier structure not as expected.")
            
        self.backbone = effnet_backbone

        # ---- Freezing Logic Removed ----
        # Freezing specific blocks in EfficientNet is less standard than ResNet.
        # Fine-tuning the entire network or using differential LR is common.
        if freeze_early_layers:
             print("Warning: freeze_early_layers=True is not actively implemented for EfficientNetV2 backbone in this version. Fine-tuning all layers.")
        # --------------------------------

        # Recreate the same output layers, now connected to EfficientNet features
        self.fc_rot = nn.Linear(num_features, 4)  # Quaternion output
        self.fc_trans = nn.Linear(num_features, 3)  # Translation output

    def forward(self, x):
        features = self.backbone(x)
        rot = self.fc_rot(features)
        trans = self.fc_trans(features)
        # Normalize quaternion to unit length for a valid rotation
        norm = rot.norm(p=2, dim=1, keepdim=True)
        rot = rot / (norm + 1e-8) # Add epsilon for numerical stability
        return rot, trans

# Keep the commented-out complex version if needed for reference
# class PoseNetComplex(nn.Module): ...