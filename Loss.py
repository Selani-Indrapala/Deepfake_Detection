import torch
import torch.nn as nn

# Define the weighted BCE loss
class WeightedBCELoss(nn.Module):
    def __init__(self, weight_bona_fide, weight_spoofed):
        super(WeightedBCELoss, self).__init__()
        self.weight_bona_fide = weight_bona_fide
        self.weight_spoofed = weight_spoofed

    def forward(self, predictions, targets):
        loss = -self.weight_bona_fide * targets * torch.log(predictions) \
               - self.weight_spoofed * (1 - targets) * torch.log(1 - predictions)
        return loss.mean()

# Initialize weights
weight_bona_fide = 4.91
weight_spoofed = 0.56

