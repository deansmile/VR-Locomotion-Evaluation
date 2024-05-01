from scipy.io import loadmat
import os

def get_file_paths(directory, exclude_files=None):
    # If no specific exclusions are provided, use a default empty list
    if exclude_files is None:
        exclude_files = []

    # Create a list to store file paths
    file_paths = []
    
    # Walk through all the directories and files in the specified directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file not in exclude_files:
                # Join the two strings to form the full filepath.
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
    
    return file_paths

# Specify the directory path
directory_path = 'ExtractedFeatures_1s'

# List of file names to exclude
excluded_files = ['label.mat', 'readme.txt']

# Get all file paths, excluding specific files
all_file_paths = get_file_paths(directory_path, exclude_files=excluded_files)
de=[]
labels=[]
label=[1,0,-1,-1,0,1,-1,0,1,1,0,-1,0,1,-1]
for fp in all_file_paths:
    # Replace 'file.mat' with the path to your .mat file
    data = loadmat(fp)
    for i in range(1,16):
        de.append(data['de_movingAve'+str(i)])
    labels.extend(label)

import h5py
with h5py.File('data_3.h5', 'w') as f:
    # Create a group for DE arrays
    DE_group = f.create_group('DE')
    
    # Since DE arrays are of different shapes, we save them individually in the group
    for i, array in enumerate(de):
        DE_group.create_dataset(str(i), data=array)
    
    # Save labels
    f.create_dataset('labels', data=labels)
