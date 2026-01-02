import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models

from .dataset import SockTripletDataset, SockPairDataset
from .model import SockProjectionHead, SockEmbeddingModel, create_projection_head

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_resnet_backbone(device: torch.device) -> nn.Module:
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet.fc = nn.Identity()  # Remove classification head
    resnet.eval()
    resnet = resnet.to(device)
    return resnet

def compute_triplet_accuracy(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> float:
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for anchor, positive, negative in dataloader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            
            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)
            
            # Cosine similarity (embeddings are L2 normalized)
            pos_sim = (anchor_emb * positive_emb).sum(dim=1)
            neg_sim = (anchor_emb * negative_emb).sum(dim=1)
            
            correct += (pos_sim > neg_sim).sum().item()
            total += anchor.size(0)
    
    return correct / total if total > 0 else 0.0

def compute_pair_accuracy(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5
) -> float:
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for img1, img2, labels in dataloader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            labels = labels.to(device)
            
            emb1 = model(img1)
            emb2 = model(img2)
            
            # Cosine similarity
            similarity = (emb1 * emb2).sum(dim=1)
            predictions = (similarity > threshold).long()
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    return correct / total if total > 0 else 0.0

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for anchor, positive, negative in dataloader:
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
        
        optimizer.zero_grad()
        
        anchor_emb = model(anchor)
        positive_emb = model(positive)
        negative_emb = model(negative)
        
        loss = criterion(anchor_emb, positive_emb, negative_emb)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches if n_batches > 0 else 0.0

def train(
    data_dir: str,
    output_path: str,
    epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    margin: float = 0.3,
    triplets_per_epoch: int = 500,
    patience: int = 15,
):
    device = get_device()
    print(f"Using device: {device}")
    
    # Create datasets
    print(f"\nLoading training data from: {data_dir}")
    train_dataset = SockTripletDataset(
        data_dir=data_dir,
        triplets_per_epoch=triplets_per_epoch,
        hard_negative_ratio=0.7
    )
    
    val_dataset = SockPairDataset(
        data_dir=data_dir,
        pairs_per_epoch=200
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Create model
    print("\nLoading ResNet50 backbone...")
    resnet = load_resnet_backbone(device)
    projection_head = create_projection_head().to(device)
    model = SockEmbeddingModel(resnet, projection_head)
    
    # Loss and optimizer
    criterion = nn.TripletMarginLoss(margin=margin, p=2)
    optimizer = optim.Adam(projection_head.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Training loop
    best_accuracy = 0.0
    epochs_without_improvement = 0
    
    print(f"\nStarting training for {epochs} epochs...")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Margin: {margin}")
    print(f"  Triplets per epoch: {triplets_per_epoch}")
    print()
    
    for epoch in range(epochs):
        # Regenerate triplets each epoch for variety
        train_dataset.regenerate_triplets()
        
        # Train
        avg_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Evaluate
        triplet_acc = compute_triplet_accuracy(model, train_loader, device)
        pair_acc = compute_pair_accuracy(model, val_loader, device)
        
        # Learning rate scheduling
        scheduler.step(pair_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1:3d}/{epochs}: loss={avg_loss:.4f}, triplet_acc={triplet_acc:.3f}, pair_acc={pair_acc:.3f}, lr={current_lr:.6f}")
        
        # Save best model
        if pair_acc > best_accuracy:
            best_accuracy = pair_acc
            epochs_without_improvement = 0
            
            # Ensure output directory exists
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save only the projection head weights
            torch.save({
                'model_state_dict': projection_head.state_dict(),
                'input_dim': 2048,
                'hidden_dim': 512,
                'output_dim': 128,
                'best_accuracy': best_accuracy,
                'epoch': epoch + 1,
            }, output_path)
            print(f"  -> Saved best model (pair_acc={best_accuracy:.3f})")
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping after {patience} epochs without improvement")
            break
    
    print(f"\nTraining complete!")
    print(f"Best pair accuracy: {best_accuracy:.3f}")
    print(f"Model saved to: {output_path}")
    
    return best_accuracy

def main():
    parser = argparse.ArgumentParser(description="Train sock embedding projection head")
    parser.add_argument("--data", type=str, default="testing/data/socks",
                        help="Path to sock training data directory")
    parser.add_argument("--output", type=str, default="server/models/sock_projection_head.pt",
                        help="Path to save trained model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--margin", type=float, default=0.3, help="Triplet loss margin")
    parser.add_argument("--triplets", type=int, default=500, help="Triplets per epoch")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    
    args = parser.parse_args()
    
    train(
        data_dir=args.data,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        margin=args.margin,
        triplets_per_epoch=args.triplets,
        patience=args.patience,
    )

if __name__ == "__main__":
    main()
