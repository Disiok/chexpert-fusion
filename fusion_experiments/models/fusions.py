import torch
from torch import nn

class VolumetricFusion(nn.Module):
    def __init__(self, num_input_features):
        super(VolumetricFusion, self).__init__()

        self.net = nn.Sequential(
            nn.BatchNorm3d(num_input_features * 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_input_features * 2, num_input_features, kernel_size=3, stride=1, padding=1, bias=False),
        )
    
    def to_volumetric(self, frontal_features, lateral_features):
        """
        Construct feature volume from frontal and lateral features

        Args:
            frontal_features (torch.FloatTensor): Frontal features of shape [B, C, H, W]
            lateral_features (torch.FloatTensor): Lateral features of shape [B, C, H, D]

        Returns:
            feature_volume (torch.FloatTensor): Feature volume of shape [B, C * 2, D, H, W]
        """
        B, C, H, W = frontal_features.shape
        B, C, H, D = lateral_features.shape

        # [B, C, D, H, W]
        frontal_features_3D = frontal_features.unsqueeze(2).expand(-1, -1, D, -1, -1)
        
        # relative to front view: [B, C, W, H, D]
        lateral_features_3D = lateral_features.unsqueeze(2).expand(-1, -1, W, -1, -1)
        
        # transpose to orient it: [B, C, D, H, W]
        transposed_lateral = lateral_features_3D.permute(0, 1, 4, 3, 2)

        # Concat on channel dimension
        concat_features = torch.cat((frontal_features_3D, transposed_lateral), dim=1)

        return concat_features
    
    def to_multiview(self, feature_volume):
        """
        Flatten feature volume back to frontal and lateral views

        Args:
            feature_volume (torch.FloatTensor): Feature volume of shape [B, C, D, H, W]

        Returns:
            frontal_features (torch.FloatTensor): Frontal features of shape [B, C, H, W]
            lateral_features (torch.FloatTensor): Lateral features of shape [B, C, H, D]
        """
        frontal_out = feature_volume.mean(2) # mean over D: [B, C, H, W]
        lateral_out = feature_volume.mean(4) # mean over W: [B, C, D, H]

        lateral_out = lateral_out.permute(0, 1, 3, 2) # back to [B, C, H, D]

        return frontal_out, lateral_out

        
    def forward(self, frontal_features, lateral_features):
        """
        Construct feature volume, apply 3D convolution, 
        and flatten back to frontal and lateral views

        Args:
            frontal_features (torch.FloatTensor): Frontal features of shape [B, C, H, W]
            lateral_features (torch.FloatTensor): Lateral features of shape [B, C, H, D]

        Returns:
            frontal_out (torch.FloatTensor): Frontal features of shape [B, C, H, W]
            lateral_out (torch.FloatTensor): Lateral features of shape [B, C, H, D]
        """
        feature_volume = self.to_volumetric(frontal_features, lateral_features) # [B, C * 2, D, H, W]
        out_volume = self.net(feature_volume) # [B, C, D, H, W]
        frontal_out, lateral_out = self.to_multiview(out_volume)
        return frontal_out, lateral_out
