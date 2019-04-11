import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict


__all__ = [
    "VolumetricFusion",
    "VolumetricFusion2D",
    "CrossSectionalFusion",
    "CrossSectionalAttentionFusion",
    "CrossSectionalAttentionFusionCorrelation",
    "CrossSectionalAttentionFusionMLP",
]


def _make_conv_block(inputs,
                     outputs,
                     kernel_size,
                     padding=0):
    """



    """
    block = nn.Sequential(
        nn.BatchNorm2d(inputs),
        nn.ReLU(inplace=True),
        nn.Conv2d(inputs, outputs, kernel_size=kernel_size, padding=padding, bias=False))
    return block


def _make_mlp(inputs, outputs):
    """



    """
    block = nn.Sequential(
        nn.ReLU(inplace=True),
        nn.Linear(inputs, outputs))
    return block


class VolumetricFusion(nn.Module):
    def __init__(self, num_input_features, is_big=False):
        super(VolumetricFusion, self).__init__()

        if is_big:
            self.net = nn.Sequential(
                nn.BatchNorm3d(num_input_features * 2),
                nn.ReLU(inplace=True),
                nn.Conv3d(num_input_features * 2, num_input_features, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm3d(num_input_features),
                nn.ReLU(inplace=True),
                nn.Conv3d(num_input_features, num_input_features, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm3d(num_input_features),
                nn.ReLU(inplace=True),
                nn.Conv3d(num_input_features, num_input_features, kernel_size=3, stride=1, padding=1, bias=False),
            )
        else:
            self.net = nn.Sequential(
                nn.BatchNorm3d(num_input_features * 2),
                nn.ReLU(inplace=True),
                nn.Conv3d(num_input_features * 2, num_input_features, kernel_size=3, stride=1, padding=1, bias=False),
            )
    
    def _to_volumetric(self, frontal_features, lateral_features):
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
    
    def _to_multiview(self, feature_volume):
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
        feature_volume = self._to_volumetric(frontal_features, lateral_features) # [B, C * 2, D, H, W]
        out_volume = self.net(feature_volume) # [B, C, D, H, W]
        frontal_out, lateral_out = self._to_multiview(out_volume)
        return frontal_out, lateral_out


class VolumetricFusion2D(nn.Module):
    def __init__(self, num_input_features, feature_size=80):
        super(VolumetricFusion2D, self).__init__()

        self.frontal_net = nn.Sequential(
            nn.BatchNorm2d(num_input_features * 2 * feature_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features * 2 * feature_size, num_input_features, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.lateral_net = nn.Sequential(
            nn.BatchNorm2d(num_input_features * 2 * feature_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features * 2 * feature_size, num_input_features, kernel_size=3, stride=1, padding=1, bias=False),
        )
    
    def _to_volumetric(self, frontal_features, lateral_features):
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

    def _to_multiview(self, feature_volume):
        """
        Flatten feature volume back to frontal and lateral views

        Args:
            feature_volume (torch.FloatTensor): Feature volume of shape [B, C, D, H, W]

        Returns:
            frontal_features (torch.FloatTensor): Frontal features of shape [B, C * D, H, W]
            lateral_features (torch.FloatTensor): Lateral features of shape [B, C * W, H, D]
        """
        B, C, D, H, W = feature_volume.shape

        frontal_view = feature_volume.view(B, C * D, H, W)
        lateral_view = feature_volume.permute(0, 1, 4, 3, 2).contiguous().view(B, C * W, H, D)

        return frontal_view, lateral_view
        
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

        feature_volume = self._to_volumetric(frontal_features, lateral_features) # [B, C * 2, D, H, W]
        frontal_view, lateral_view = self._to_multiview(feature_volume)

        frontal_out = self.frontal_net(frontal_view)
        lateral_out = self.lateral_net(lateral_view)

        return frontal_out, lateral_out


class CrossSectionalFusion(nn.Module):
    def __init__(self, num_input_features):
        super(CrossSectionalFusion, self).__init__()

        self.frontal_net = nn.Sequential(
            nn.BatchNorm2d(num_input_features * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features * 2, num_input_features, kernel_size=1, stride=1, bias=False)
        )

        self.lateral_net = nn.Sequential(
            nn.BatchNorm2d(num_input_features * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features * 2, num_input_features, kernel_size=1, stride=1, bias=False)
        )
    
    def forward(self, frontal_features, lateral_features):
        B, C, H, W = frontal_features.shape
        B, C, H, D = lateral_features.shape


        frontal_feature_column = F.adaptive_avg_pool2d(frontal_features, (H, 1))
        lateral_feature_column = F.adaptive_avg_pool2d(lateral_features, (H, 1))

        frontal_transfer = frontal_feature_column.expand(B, C, H, D)
        lateral_transfer = lateral_feature_column.expand(B, C, H, W)

        frontal_features = torch.cat((frontal_features, lateral_transfer), dim=1)
        lateral_features = torch.cat((lateral_features, frontal_transfer), dim=1)

        frontal_features = self.frontal_net(frontal_features)
        lateral_features = self.lateral_net(lateral_features)

        return frontal_features, lateral_features


class CrossSectionalAttentionFusion(nn.Module):
     def __init__(self, num_input_features, width):
         super(CrossSectionalAttentionFusion, self).__init__()

         self.frontal_net = nn.Conv2d(num_input_features * 2, num_input_features, kernel_size=1, stride=1, bias=False)
         self.lateral_net = nn.Conv2d(num_input_features * 2, num_input_features, kernel_size=1, stride=1, bias=False)

         self.frontal_net_attention_mask = nn.Sequential(
             nn.Conv2d(num_input_features * 2, 32, kernel_size=1, stride=1, bias=False),
             nn.BatchNorm2d(32),
             nn.ReLU(inplace=True),
             nn.Conv2d(32, 1, kernel_size=1, stride=1, bias=False),
             nn.ReLU(inplace=True),
         )

         self.lateral_net_attention_mask = nn.Sequential(
             nn.Conv2d(num_input_features * 2, 32, kernel_size=1, stride=1, bias=False),
             nn.BatchNorm2d(32),
             nn.ReLU(inplace=True),
             nn.Conv2d(32, 1, kernel_size=1, stride=1, bias=False),
             nn.ReLU(inplace=True),
         )

     def forward(self, frontal_features, lateral_features):
         B, C, H, W = frontal_features.shape
         B, C, H, D = lateral_features.shape

         frontal_mask = self.frontal_net_attention_mask(torch.cat((frontal_features, lateral_features), dim=1))
         lateral_mask = self.lateral_net_attention_mask(torch.cat((frontal_features, lateral_features), dim=1))

         frontal_mask = F.softmax(frontal_mask, dim=-1)
         lateral_mask = F.softmax(lateral_mask, dim=-1)

         frontal_transfer = frontal_mask * lateral_features
         lateral_transfer = lateral_mask * frontal_features

         frontal_features = torch.cat((frontal_features, lateral_transfer), dim=1)
         lateral_features = torch.cat((lateral_features, frontal_transfer), dim=1)

         frontal_features = self.frontal_net(frontal_features)
         lateral_features = self.lateral_net(lateral_features)

         return frontal_features, lateral_features


class CrossSectionalAttentionFusionCorrelation(nn.Module):
    def __init__(self, num_input_features, width):
        super(CrossSectionalAttentionFusionCorrelation, self).__init__()

        self.frontal_net = nn.Conv2d(num_input_features * 2, num_input_features, kernel_size=1, stride=1, bias=False)
        self.lateral_net = nn.Conv2d(num_input_features * 2, num_input_features, kernel_size=1, stride=1, bias=False)

        self.frontal_net_attention_mask = nn.Sequential(
            nn.Conv2d(num_input_features * 2, 32, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
        )

        self.lateral_net_attention_mask = nn.Sequential(
            nn.Conv2d(num_input_features * 2, 32, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, frontal_features, lateral_features):
        B, C, H, W = frontal_features.shape

        front_mask = torch.zeros([B, 1, H, W, W], dtype=torch.float32).cuda()
        lateral_transfer = torch.zeros([B, C, H, W], dtype=torch.float32).cuda()

        for ii in range(W):
            for jj in range(W):
                front_mask[:, :, :, ii, jj] = (frontal_features[:, :, :, ii] * lateral_features[:, :, :, jj]).norm()

        front_mask = F.softmax(front_mask, dim=-1)

        for ii in range(H):
            for jj in range(W):
                lateral_transfer[:, :, ii, jj] = (lateral_features[:, :, ii, :] * front_mask[:, :, ii, jj, :]).sum(dim=-1)

        frontal_features = torch.cat((frontal_features, lateral_transfer), dim=1)
        frontal_features = self.frontal_net(frontal_features)
        return frontal_features, lateral_features


class CrossSectionalAttentionFusionMLP(nn.Module):
    def __init__(self, num_input_features, width):
        super(CrossSectionalAttentionFusionMLP, self).__init__()

        self.frontal_net = _make_conv_block(num_input_features * 2, num_input_features, 1)
        self.lateral_net = _make_conv_block(num_input_features * 2, num_input_features, 1)

        self.frontal_attention_conv = nn.Sequential(
            _make_conv_block(num_input_features * 2, 32, 1),
            _make_conv_block(32, 8, 1),
        )
        self.frontal_attention_mlp = _make_mlp(width * 8, width * width)

        self.lateral_attention_conv = nn.Sequential(
            _make_conv_block(num_input_features * 2, 32, 1),
            _make_conv_block(32, 8, 1),
        )
        self.lateral_attention_mlp = _make_mlp(width * 8, width * width)

    def forward(self, frontal_features, lateral_features):
        B, C, H, W = frontal_features.shape
        B, C, H, D = lateral_features.shape

        frontal_mask = torch.cat((frontal_features, lateral_features), dim=1)
        frontal_mask = self.frontal_attention_conv(frontal_mask)
        frontal_mask = frontal_mask.transpose(1, 2).contiguous().view(B, H, -1)
        frontal_mask = self.frontal_attention_mlp(frontal_mask).view(B, 1, H, W, W)
        frontal_mask = F.softmax(frontal_mask, dim=-1)

        lateral_mask = torch.cat((lateral_features, frontal_features), dim=1)
        lateral_mask = self.lateral_attention_conv(lateral_mask)
        lateral_mask = lateral_mask.transpose(1, 2).contiguous().view(B, H, -1)
        lateral_mask = self.lateral_attention_mlp(lateral_mask).view(B, 1, H, W, W)
        lateral_mask = F.softmax(lateral_mask, dim=-1)

        frontal_transfer = frontal_mask * lateral_features.unsqueeze(-1)
        frontal_transfer = torch.sum(frontal_transfer, dim=-1)

        lateral_transfer = lateral_mask * frontal_features.unsqueeze(-1)
        lateral_transfer = torch.sum(lateral_transfer, dim=-1)

        frontal_features = torch.cat((frontal_features, lateral_transfer), dim=1)
        lateral_features = torch.cat((lateral_features, frontal_transfer), dim=1)

        frontal_features = self.frontal_net(frontal_features)
        lateral_features = self.lateral_net(lateral_features)

        return frontal_features, lateral_features
