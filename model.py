import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_V2_S_Weights, ResNet50_Weights, ResNet34_Weights

# Model Definition: Uses EfficientNetV2-S as backbone and appends quaternion and translation heads to the end of the network
class PoseNet(nn.Module):
    def __init__(self, architecture='efficientnet_v2_s', pretrained=True, freeze_early_backbone_layers=True):
        super(PoseNet, self).__init__()
        self.architecture = architecture
        self.freeze_early_backbone_layers = freeze_early_backbone_layers
        print(f"POSENET: Architecture: {self.architecture}, Freeze Early Backbone Layers: {self.freeze_early_backbone_layers}")

        if self.architecture == 'efficientnet_v2_s':
            effnet_weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
            effnet_backbone = models.efficientnet_v2_s(weights=effnet_weights)
            
            if self.freeze_early_backbone_layers:
                # Freeze the first 2 blocks of features in EfficientNetV2-S
                # effnet_backbone.features is a Sequential module
                if len(effnet_backbone.features) >= 2:
                    for i in range(2):
                        for param in effnet_backbone.features[i].parameters():
                            param.requires_grad = False
                    print("Froze EfficientNetV2-S features blocks 0 and 1.")
                else:
                    print("Warning: EfficientNetV2-S features has less than 2 blocks, could not freeze as intended.")

            if isinstance(effnet_backbone.classifier, nn.Sequential) and len(effnet_backbone.classifier) > 0:
                final_layer = effnet_backbone.classifier[-1]
                if isinstance(final_layer, nn.Linear):
                    num_features = final_layer.in_features
                    effnet_backbone.classifier = nn.Identity()
                else:
                    raise TypeError("EfficientNetV2 classifier's last layer is not Linear.")
            else:
                raise TypeError("EfficientNetV2 classifier structure not as expected.")
            
            self.backbone = effnet_backbone
            self.fc_rot = nn.Linear(num_features, 4)  # Quaternion output
            self.fc_trans = nn.Linear(num_features, 3)  # Translation output

        elif self.architecture == 'resnet34':
            resnet_weights = ResNet34_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet34(weights=resnet_weights)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity() # Remove original classifier

            if self.freeze_early_backbone_layers:
                # Freeze conv1, bn1, layer1 for ResNet architectures
                for name, param in self.backbone.named_parameters():
                    if name.startswith('conv1') or name.startswith('bn1') or \
                       name.startswith('layer1'): # layer1 for ResNet
                        param.requires_grad = False
                print("Froze ResNet34 layers: conv1, bn1, layer1")
            
            # Simplified head for ResNet34
            self.fc_rot = nn.Linear(num_features, 4)  # Quaternion output
            self.fc_trans = nn.Linear(num_features, 3)  # Translation output

        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}. Choose 'efficientnet_v2_s' or 'resnet34'.")

    def forward(self, x):
        features = self.backbone(x)
        # Common simple head for both architectures now
        rot = self.fc_rot(features)
        trans = self.fc_trans(features)

        # Normalize quaternion to unit length for a valid rotation
        norm = rot.norm(p=2, dim=1, keepdim=True)
        rot = rot / (norm + 1e-8) # Add epsilon for numerical stability
        return rot, trans