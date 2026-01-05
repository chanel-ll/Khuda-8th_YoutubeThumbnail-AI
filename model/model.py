import torch
import torch.nn as nn
import clip

class CLIPResNet50Regressor(nn.Module):
    def __init__(self, device='cuda'):
        super(CLIPResNet50Regressor, self).__init__()
       
        self.clip_model, _ = clip.load("RN50", device=device, jit=False)
        
        self.visual_encoder = self.clip_model.visual

        visual_dim = 1024 
        intensity_dim = 1   
        
        self.regressor = nn.Sequential(
            nn.Linear(visual_dim + intensity_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128, 1),
            nn.Sigmoid() 
        )

        self.visual_encoder = self.visual_encoder.float()

    def forward(self, images, intensities):

        visual_features = self.visual_encoder(images)
        visual_features = visual_features.float() 

        combined = torch.cat((visual_features, intensities), dim=1)
    
        output = self.regressor(combined)
        return output