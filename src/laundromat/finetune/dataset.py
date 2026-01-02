import os
import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class SockTripletDataset(Dataset):

    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[T.Compose] = None,
        triplets_per_epoch: int = 500,
        hard_negative_ratio: float = 0.7,
    ):
        self.data_dir = Path(data_dir)
        self.triplets_per_epoch = triplets_per_epoch
        self.hard_negative_ratio = hard_negative_ratio
        
        if transform is None:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.RandomHorizontalFlip(),
                T.RandomRotation(15),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transform
        
        # Load sock data: {sock_id: {'color': str, 'images': [paths]}}
        self.socks = {}
        self.color_groups = {}  # {color: [sock_ids]}
        self._load_socks()
        
        # Pre-generate triplets for this epoch
        self.triplets = self._generate_triplets()
    
    def _load_socks(self):
        for color_dir in self.data_dir.iterdir():
            if not color_dir.is_dir():
                continue
            color = color_dir.name  # 'grey' or 'white'
            
            if color not in self.color_groups:
                self.color_groups[color] = []
            
            for sock_dir in color_dir.iterdir():
                if not sock_dir.is_dir():
                    continue
                sock_id = f"{color}/{sock_dir.name}"
                
                images = []
                for img_path in sock_dir.iterdir():
                    if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                        images.append(str(img_path))
                
                if len(images) >= 2:  # Need at least 2 images to form positive pairs
                    self.socks[sock_id] = {
                        'color': color,
                        'images': images
                    }
                    self.color_groups[color].append(sock_id)
        
        print(f"Loaded {len(self.socks)} socks:")
        for color, sock_ids in self.color_groups.items():
            print(f"  {color}: {len(sock_ids)} socks")
    
    def _generate_triplets(self) -> List[Tuple[str, str, str]]:
        triplets = []
        sock_ids = list(self.socks.keys())
        
        for _ in range(self.triplets_per_epoch):
            # Pick anchor sock
            anchor_sock = random.choice(sock_ids)
            anchor_images = self.socks[anchor_sock]['images']
            anchor_color = self.socks[anchor_sock]['color']
            
            # Pick anchor and positive (different images of same sock)
            anchor_img, positive_img = random.sample(anchor_images, 2)
            
            # Pick negative sock (different sock)
            use_hard_negative = random.random() < self.hard_negative_ratio
            
            if use_hard_negative and len(self.color_groups[anchor_color]) > 1:
                # Hard negative: same color, different sock
                same_color_socks = [s for s in self.color_groups[anchor_color] if s != anchor_sock]
                negative_sock = random.choice(same_color_socks)
            else:
                # Easy negative: any different sock
                other_socks = [s for s in sock_ids if s != anchor_sock]
                negative_sock = random.choice(other_socks)
            
            negative_img = random.choice(self.socks[negative_sock]['images'])
            
            triplets.append((anchor_img, positive_img, negative_img))
        
        return triplets
    
    def __len__(self) -> int:
        return len(self.triplets)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_path, positive_path, negative_path = self.triplets[idx]
        
        anchor = self._load_image(anchor_path)
        positive = self._load_image(positive_path)
        negative = self._load_image(negative_path)
        
        return anchor, positive, negative
    
    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert('RGB')
        return self.transform(img)
    
    def regenerate_triplets(self):
        self.triplets = self._generate_triplets()


class SockPairDataset(Dataset):
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[T.Compose] = None,
        pairs_per_epoch: int = 200,
    ):
        self.data_dir = Path(data_dir)
        self.pairs_per_epoch = pairs_per_epoch
        
        if transform is None:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transform
        
        self.socks = {}
        self._load_socks()
        self.pairs = self._generate_pairs()
    
    def _load_socks(self):
        for color_dir in self.data_dir.iterdir():
            if not color_dir.is_dir():
                continue
            color = color_dir.name
            
            for sock_dir in color_dir.iterdir():
                if not sock_dir.is_dir():
                    continue
                sock_id = f"{color}/{sock_dir.name}"
                
                images = []
                for img_path in sock_dir.iterdir():
                    if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                        images.append(str(img_path))
                
                if len(images) >= 2:
                    self.socks[sock_id] = images
    
    def _generate_pairs(self) -> List[Tuple[str, str, int]]:
        pairs = []
        sock_ids = list(self.socks.keys())
        
        # Half positive, half negative
        n_positive = self.pairs_per_epoch // 2
        n_negative = self.pairs_per_epoch - n_positive
        
        # Positive pairs
        for _ in range(n_positive):
            sock_id = random.choice(sock_ids)
            img1, img2 = random.sample(self.socks[sock_id], 2)
            pairs.append((img1, img2, 1))
        
        # Negative pairs
        for _ in range(n_negative):
            sock1, sock2 = random.sample(sock_ids, 2)
            img1 = random.choice(self.socks[sock1])
            img2 = random.choice(self.socks[sock2])
            pairs.append((img1, img2, 0))
        
        random.shuffle(pairs)
        return pairs
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        path1, path2, label = self.pairs[idx]
        img1 = self._load_image(path1)
        img2 = self._load_image(path2)
        return img1, img2, label
    
    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert('RGB')
        return self.transform(img)
