import torch
import torch.nn as nn
import torch.nn.functional as F

class SockProjectionHead(nn.Module):
    
    def __init__(self, input_dim: int = 2048, hidden_dim: int = 512, output_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # L2 normalize for cosine similarity
        x = F.normalize(x, p=2, dim=-1)
        return x

class SockEmbeddingModel(nn.Module):
    
    def __init__(self, resnet: nn.Module, projection_head: SockProjectionHead):
        super().__init__()
        self.resnet = resnet
        self.projection_head = projection_head
        
        # Freeze ResNet
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.eval()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.resnet(x)
        embeddings = self.projection_head(features)
        return embeddings
    
    def train(self, mode: bool = True):
        # Keep ResNet in eval mode, only train projection head
        super().train(mode)
        self.resnet.eval()
        return self

def create_projection_head(
    input_dim: int = 2048,
    hidden_dim: int = 512, 
    output_dim: int = 128,
    dropout: float = 0.1
) -> SockProjectionHead:
    return SockProjectionHead(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout=dropout
    )
