import torch
from torch.utils.data import Sampler
import random

class CustomBatchSampler(Sampler):
    def __init__(self, labels, batch_size):
        self.batch_size = batch_size
        self.labels = labels
        self.indices_by_class = self._group_by_class()
        self.all_indices = list(range(len(labels)))
        self.num_samples = len(labels)

    def _group_by_class(self):
        """Organizes indices by class for balanced sampling."""
        indices_by_class = {}
        for idx, label in enumerate(self.labels):
            if label not in indices_by_class:
                indices_by_class[label] = []
            indices_by_class[label].append(idx)
        return indices_by_class

    def __iter__(self):
        
        used_indices = set()
        batches = []
        iteration = 0  # Debug counter
        
        while len(used_indices) < self.num_samples:
            iteration += 1
            if iteration % 1000 == 0:
                print(f"Iteration: {iteration}, Used indices: {len(used_indices)}/{self.num_samples}")

            # If fewer than batch_size indices remain, exit to prevent infinite loop
            remaining = self.num_samples - len(used_indices)
            if remaining < self.batch_size:
                print("Stopping early: Not enough samples left for a full batch.")
                break

            #classes = random.sample(self.indices_by_class.keys(), 4)
            classes = random.sample(list(self.indices_by_class.keys()), 4)
            batch = []
            for cls in classes:
                available = list(set(self.indices_by_class[cls]) - used_indices)
                if len(available) >= 2:
                    selected = random.sample(available, 2)
                    batch.extend(selected)
                    used_indices.update(selected)
            
            if len(batch) == self.batch_size:
                batches.append(batch)
            
            # Extra safeguard against infinite loop
            if iteration > 100000:
                print("ERROR: iterations exceeded 100000, exiting...")
                break

        random.shuffle(batches)
        return iter(batches)


    def __len__(self):
        return self.num_samples // self.batch_size

# Assuming dataset_train is created using original SSVideoClsDataset
#dataset_train = SSVideoClsDataset(anno_path="path/to/annotations.txt", data_path="path/to/data", mode='train')
#sampler_train = CustomBatchSampler(dataset_train.label_array, batch_size=8)


