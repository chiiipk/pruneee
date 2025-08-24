import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List, Optional
import json
import random

class ContrastiveDataset(Dataset):
    """Contrastive learning dataset for embedding training"""
    
    def __init__(self, data_path: str, tokenizer_name: str = "BAAI/bge-m3", max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.data = self._load_data(data_path)
    
    def _load_data(self, data_path: str) -> List[Dict]:
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                data.append({
                    'anchor': item['anchor'],
                    'positive': item['positive'],
                    'negative': item.get('negative', [])
                })
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Encode anchor
        anchor_encoded = self.tokenizer(
            item['anchor'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Encode positive
        positive_encoded = self.tokenizer(
            item['positive'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': torch.cat([anchor_encoded['input_ids'], positive_encoded['input_ids']], dim=0).squeeze(1),
            'attention_mask': torch.cat([anchor_encoded['attention_mask'], positive_encoded['attention_mask']], dim=0).squeeze(1),
            'labels': torch.tensor([1], dtype=torch.long)  # Positive pair label
        }

def create_contrastive_dataloader(data_path: str, batch_size: int = 32, tokenizer_name: str = "BAAI/bge-m3",
                                 max_length: int = 512, shuffle: bool = True) -> DataLoader:
    """Create contrastive learning dataloader"""
    dataset = ContrastiveDataset(data_path, tokenizer_name, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
