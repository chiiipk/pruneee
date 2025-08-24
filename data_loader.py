"""
Unified HuggingFace Dataset Loader for BGE-M3 Pruning
Clean, minimal implementation using datasets library
"""

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
# Import HuggingFace datasets explicitly to avoid local module conflict
import datasets
from typing import Dict, Any, Optional

class BGEDataset(torch.utils.data.Dataset):
    """Unified dataset for BGE-M3 training with HuggingFace datasets"""
    
    def __init__(self, dataset_name: str, split: str = "train", 
                 tokenizer_name: str = "BAAI/bge-m3", max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
        # Load dataset from HuggingFace
        if dataset_name == "sts":
            self.dataset = datasets.load_dataset("mteb/stsbenchmark-sts", split=split)
            self.task_type = "similarity"
        elif dataset_name == "msmarco":
            self.dataset = datasets.load_dataset("ms_marco", "v1.1", split=split)
            self.task_type = "retrieval"
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.dataset[idx]
        
        if self.task_type == "similarity":
            # STS format: sentence1, sentence2, score
            inputs1 = self.tokenizer(
                item['sentence1'], max_length=self.max_length,
                padding='max_length', truncation=True, return_tensors='pt'
            )
            inputs2 = self.tokenizer(
                item['sentence2'], max_length=self.max_length,
                padding='max_length', truncation=True, return_tensors='pt'
            )
            
            return {
                'input_ids': torch.cat([inputs1['input_ids'], inputs2['input_ids']], dim=0),
                'attention_mask': torch.cat([inputs1['attention_mask'], inputs2['attention_mask']], dim=0),
                'similarity_scores': torch.tensor(item['score'], dtype=torch.float),
                'task_type': 'sts'
            }
        
        else:  # retrieval
            # MS MARCO format: query, passage
            query_inputs = self.tokenizer(
                item['query'], max_length=self.max_length,
                padding='max_length', truncation=True, return_tensors='pt'
            )
            passage_inputs = self.tokenizer(
                item['passage'], max_length=self.max_length,
                padding='max_length', truncation=True, return_tensors='pt'
            )
            
            return {
                'input_ids': torch.cat([query_inputs['input_ids'], passage_inputs['input_ids']], dim=0),
                'attention_mask': torch.cat([query_inputs['attention_mask'], passage_inputs['attention_mask']], dim=0),
                'labels': torch.tensor(1, dtype=torch.long),  # Positive pair
                'task_type': 'retrieval'
            }

def create_dataloader(dataset_name: str, split: str = "train", batch_size: int = 16, 
                     tokenizer_name: str = "BAAI/bge-m3", max_length: int = 512,
                     num_workers: int = 2) -> DataLoader:
    """Create DataLoader for BGE-M3 training"""
    dataset = BGEDataset(dataset_name, split, tokenizer_name, max_length)
    
    def collate_fn(batch):
        """Custom collate function"""
        input_ids = torch.stack([item['input_ids'].squeeze(0) for item in batch])
        attention_mask = torch.stack([item['attention_mask'].squeeze(0) for item in batch])
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'task_type': batch[0]['task_type']
        }
        
        if batch[0]['task_type'] == 'sts':
            result['similarity_scores'] = torch.stack([item['similarity_scores'] for item in batch])
        else:
            result['labels'] = torch.stack([item['labels'] for item in batch])
        
        return result
    
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=(split == "train"),
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
    )
