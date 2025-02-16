from torch.utils.data import DataLoader, Sampler
from collections import defaultdict
import random

# Load dataset
from dataset import CustomImageDataset
from config import load_config

class GroupedSampler(Sampler):
    """
    Custom Sampler that groups images by their unique_id.
    Ensures that each batch contains images from only one unique_id.
    """

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.grouped_indices = self._group_by_id()

    def _group_by_id(self):
        """
        Groups dataset indices by unique_id.
        """
        groups = defaultdict(list)
        for idx, sample in enumerate(self.dataset):
            groups[sample["unique_id"]].append(idx)
        return list(groups.values())

    def __iter__(self):
        """
        Creates batches that contain only one unique_id per batch.
        """
        batch_list = []

        # Shuffle groups to ensure randomness
        random.shuffle(self.grouped_indices)

        for group in self.grouped_indices:
            random.shuffle(group)  # Shuffle images within each ID group
            for i in range(0, len(group), self.batch_size):
                batch = group[i : i + self.batch_size]
                batch_list.append(batch)

        random.shuffle(batch_list)  # Shuffle batches for training randomness
        return iter(batch_list)

    def __len__(self):
        """
        Returns the number of batches.
        """
        return sum(len(group) // self.batch_size + (1 if len(group) % self.batch_size != 0 else 0) for group in self.grouped_indices)


def custom_collate_fn(batch):
    """
    Splits batch into multiple mini-batches if different unique_ids are present.
    
    Args:
        batch (list): List of dataset samples.

    Returns:
        list of dicts: Each dict contains "unique_id", "image", and "tensor" keys.
    """
    grouped_batches = defaultdict(list)
    
    for item in batch:
        grouped_batches[item["unique_id"]].append(item)

    return list(grouped_batches.values())  # Return separate batches per unique_id


if __name__ == "__main__": 
    cfg = load_config()
    dataset = CustomImageDataset(cfg)

    # Create DataLoader with the custom sampler
    batch_size = cfg['batch_size']
    sampler = GroupedSampler(dataset, batch_size)
    dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=custom_collate_fn)

    # Example usage:
    for mini_batches in dataloader:
        for batch in mini_batches:
            print(f"Batch with ID {batch[0]['unique_id']} - Batch size: {len(batch)}")
