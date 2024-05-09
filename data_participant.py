from scipy.io import loadmat
import os
import h5py

def get_file_paths(directory, test_prefixes=None, excluded_files=None):
    if test_prefixes is None:
        test_prefixes = []
    if excluded_files is None:
        excluded_files = []

    train_files = []
    test_files = []
    
    # Walk through all the directories and files in the specified directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.mat') and file not in excluded_files:
                if any(file.startswith(prefix) for prefix in test_prefixes):
                    test_files.append(os.path.join(root, file))
                else:
                    train_files.append(os.path.join(root, file))
    
    return train_files, test_files

# Specify the directory path, test prefixes, and excluded files
directory_path = "E:\\SEED Dataset\\ExtractedFeatures_1s\\ExtractedFeatures_1s"
test_prefixes = ['13', '14', '15']
excluded_files = ['label.mat', 'readme.txt']

# Get train and test file paths
train_file_paths, test_file_paths = get_file_paths(directory_path, test_prefixes=test_prefixes, excluded_files=excluded_files)

# Load and prepare data
def load_data(file_paths):
    de = []
    labels = []
    label = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]  # Assuming this is correct for all
    for fp in file_paths:
        data = loadmat(fp)
        for i in range(1, 16):
            de.append(data['de_movingAve' + str(i)])
        labels.extend(label)
    return de, labels

train_de, train_labels = load_data(train_file_paths)
test_de, test_labels = load_data(test_file_paths)

# Create separate HDF5 files for train and test datasets
def save_data(filename, de, labels):
    with h5py.File(filename, 'w') as f:
        DE_group = f.create_group('DE')
        for i, array in enumerate(de):
            DE_group.create_dataset(str(i), data=array)
        f.create_dataset('labels', data=labels)

save_data('train_data.h5', train_de, train_labels)
save_data('test_data.h5', test_de, test_labels)
