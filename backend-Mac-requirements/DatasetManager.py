from torch.utils.data import Dataset, Subset
from pathlib import Path
from misc_utils import converge_feature_options
import random
from collections import defaultdict
import h5py

# From: https://gist.github.com/Alvtron/9b9c2f870df6a54fda24dbd1affdc254
def stratified_split(dataset : Dataset, labels, fraction, random_state=None, force_one_sample_per_label=True):
    # Set random state if provided
    if random_state:
        random.seed(random_state)

    # Sort indices by label
    indices_per_label = defaultdict(list)
    for index, label in enumerate(labels):
        indices_per_label[label].append(index)
    first_set_indices, second_set_indices = list(), list()

    # divvy up indices between sets
    for label, indices in indices_per_label.items():
        n_samples_for_label = round(len(indices) * fraction) # get number of samples for first set
        if force_one_sample_per_label: # Force at least one sample to be selected per label, regardless of fraction
            n_samples_for_label = max(n_samples_for_label, 1) # force at least 1 sample to always be in the first set
            n_samples_for_label = min(n_samples_for_label, len(indices)-1) # force at least 1 sample to always be in the second set, i.e. at least one sample is not in the first set
        random_indices_sample = random.sample(indices, n_samples_for_label) # Random sampling
        first_set_indices.extend(random_indices_sample) # Assign selected indicies to first list
        second_set_indices.extend(set(indices) - set(random_indices_sample)) # Assign remaining indices to second list

    # Go from indicies to full datasets
    first_set_inputs = Subset(dataset, first_set_indices)
    first_set_labels = list(map(labels.__getitem__, first_set_indices))
    second_set_inputs = Subset(dataset, second_set_indices)
    second_set_labels = list(map(labels.__getitem__, second_set_indices))

    return first_set_inputs, first_set_labels, second_set_inputs, second_set_labels

## TODO: get features directory as input and not a list of files
class FeatureDataset(Dataset):
    
    def __init__(self, feature_files: list[Path], label_map = None):
        self.files = feature_files
        self.labels = [f.parent.stem for f in feature_files]

        if label_map is None:
            self.class_idx_map = {}
            for i, label in enumerate(sorted(list(set(self.labels)))): # Forms label to index map aphabetically
                self.class_idx_map[label] = i
        else:
            self.class_idx_map = label_map

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        coords = h5py.File(self.files[idx])['coords'][:] if Path(self.files[idx]).suffix == '.h5' else None # Gets coords from file if they exist
        features = converge_feature_options(self.files[idx]) # Gets features from file
        label = self.class_idx_map[self.labels[idx]] # Converts label from text to index

        return coords, features, label

if __name__ == '__main__':
    features = [Path('feature_data/tiny_camel/test/negative/normal_003.h5'), 
                Path('feature_data/tiny_camel/test//negative/normal_004.h5'), 
                Path('feature_data/tiny_camel/test/positive/tumor_003.h5'), 
                Path('feature_data/tiny_camel/test/positive/tumor_004.h5')]
    ds = FeatureDataset(features)
    ds_0, _, ds_1, _ = stratified_split(ds, ds.labels, 0.8, None)
    print(f'{len(ds)=}')
    print(f'{len(ds_0)=}')
    print(f'{len(ds_1)=}')
    print([ds.files[i] for i in ds_0.indices])
    print([ds.files[i] for i in ds_1.indices])