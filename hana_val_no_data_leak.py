import numpy as np
import os
from sklearn.model_selection import StratifiedKFold

# --- 1. Load Real Data (Relies on your provided directory structure) ---
# Please ensure this path is correct and contains 'class_0' and 'class_1' directories
data_dir = '/path/to/the/dataset' 
all_image_paths = []
all_labels = []
# Assuming you have two classes as per your original script
label_map = {'class_0': 0, 'class_1': 1} 

print(f"--- Loading data from directory '{data_dir}'... ---")

# Traverse the directories to collect all image paths and labels
for class_name in os.listdir(data_dir):
    if class_name in label_map:
        class_dir = os.path.join(data_dir, class_name)
        label = label_map[class_name]
        # Check if it's a directory before listing contents
        if os.path.isdir(class_dir):
            for image_name in os.listdir(class_dir):
                all_image_paths.append(os.path.join(class_dir, image_name))
                all_labels.append(label)

all_image_paths = np.array(all_image_paths)
all_labels = np.array(all_labels)

total_samples = len(all_labels)

if total_samples == 0:
    print("\nERROR: Failed to load any samples from the specified path.")
    print("Please check if data_dir is correct and if there are files in class_0/class_1 directories.")
    exit()

# Calculate initial class distribution
unique_labels, counts = np.unique(all_labels, return_counts=True)
class_counts = dict(zip(unique_labels, counts))

print(f"Total Samples: {total_samples}")
print(f"Original Class Distribution: {class_counts}")
print("-" * 30)

# --- 2. Stratified K-Fold Splitting Logic ---
n_splits = 5
# Use the same random state as your original script for consistent splitting
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Iterate through each fold
for fold, (train_index, val_index) in enumerate(skf.split(all_image_paths, all_labels)):
    
    # Get the labels for the current fold's training and validation sets
    train_labels = all_labels[train_index]
    val_labels = all_labels[val_index]

    # Get the number of samples in the validation and training sets
    num_train = len(train_labels)
    num_val = len(val_labels)
    
    # Calculate class distribution for validation and training sets
    train_counts = {label: np.sum(train_labels == label) for label in np.unique(train_labels)}
    val_counts = {label: np.sum(val_labels == label) for label in np.unique(val_labels)}
    
    # --- Cross-Validation Check for Leakage ---
    # The length of the intersection of indices must be 0, confirming no overlap.
    intersection_length = len(np.intersect1d(train_index, val_index))
    
    print(f"### Fold {fold+1}/{n_splits} ###")
    print(f"  > Training Set Samples: {num_train}")
    print(f"    - Training Set Class Distribution: {train_counts}")
    print(f"  > Validation Set Samples: {num_val}")
    print(f"    - Validation Set Class Distribution: {val_counts}")
    print(f"  > Index Overlap Check (Intersection Length): {intersection_length} (Must be 0)")
    print("-" * 30)
    
print("--- 5-Fold Split Verification Complete ---")