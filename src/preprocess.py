
"""
Handles loading of filenames and metadata.
Chromagram arrays are loaded later in prepare_dataset.
"""

import os
import pandas as pd
import numpy as np
import random

def load_required_filenames(metadata, required_ragas):
    """
    Filter filenames from metadata that belong to the specified list of ragas.

    Parameters:
    metadata (pd.DataFrame): Metadata containing 'raga name' and 'audio_file' columns
    required_ragas (list): List of raga names to filter for

    Returns:
    list: Filenames corresponding to the required ragas
    """
    # Filter metadata for only the required ragas
    filtered_metadata = metadata[metadata['raga name'].isin(required_ragas)]

    # Extract corresponding filenames
    filtered_filenames = filtered_metadata['audio_file'].tolist()

    return filtered_filenames, filtered_metadata



def load_metadata(metadata_path):
    """
    Loads the metadata CSV file and ensures column names are lowercase.

    Parameters:
    metadata_path (str): Path to the metadata CSV file

    Returns:
    pandas.DataFrame: Cleaned metadata dataframe
    """
    df = pd.read_csv(metadata_path)
    df.columns = [col.strip().lower() for col in df.columns]
    return df

def load_single_chromagram(chroma_dir, filename):
    """
    Loads a single chromagram numpy array from the given directory.

    Parameters:
    chroma_dir (str): Directory containing chromagram .npy files
    filename (str): Filename without extension

    Returns:
    np.ndarray: Loaded chromagram array
    """
    return np.load(os.path.join(chroma_dir, f"{filename}.npy"))


def create_train_val_test_splits(metadata, files_dir):
    """
    Create train, validation, and test splits using the 'raga_name' and 'audio_file' columns.

    Parameters:
        metadata (pd.DataFrame): Must have 'raga_name' and 'audio_file' columns.

    Returns:
        tuple: (train_files, validation_files, test_files)
    """
    value_counts = metadata['raga name'].value_counts()
    # print(value_counts)
    file_names=os.listdir(files_dir)

    train_files = []
    test_files = []
    validation_files = []

    for raga_name, count in value_counts.items():
        unique_names = metadata[metadata['raga name'] == raga_name]['audio_file'].tolist()

        if count < 5:
            test_files.extend(random.sample(unique_names, 1))
            remaining = [name for name in unique_names if name not in test_files]
            validation_files.extend(random.sample(remaining, 1))
            train_files.extend([name for name in unique_names if name not in test_files and name not in validation_files])
        elif 6 <= count <= 12:
            # 1 test file, 2 validation files
            test_files.extend(random.sample(unique_names, 1))
            remaining = [name for name in unique_names if name not in test_files]
            validation_files.extend(random.sample(remaining, 2))
            train_files.extend([name for name in unique_names if name not in test_files and name not in validation_files])
        elif 12 < count <= 20:
            # 2 test files, 3 validation files
            test_files.extend(random.sample(unique_names, 2))
            remaining = [name for name in unique_names if name not in test_files]
            validation_files.extend(random.sample(remaining, 3))
            train_files.extend([name for name in unique_names if name not in test_files and name not in validation_files])
        elif count > 20:
            # 4 test files, 6 validation files
            test_files.extend(random.sample(unique_names, 4))
            remaining = [name for name in unique_names if name not in test_files]
            validation_files.extend(random.sample(remaining, 6))
            train_files.extend([name for name in unique_names if name not in test_files and name not in validation_files])

    train_files = list(set(train_files))
    test_files = list(set(test_files))
    validation_files = list(set(validation_files))
    return train_files,validation_files,test_files



def match_filenames_by_prefix(all_filenames, prefix_list):
    """
    Match filenames that start with a prefix from the given list.

    Parameters:
    all_filenames (list): List of all filenames in the folder
    prefix_list (list): List of unique_names to match as prefixes

    Returns:
    list: Matched filenames
    """
    matched_files = []
    for prefix in prefix_list:
        matched = [fname for fname in all_filenames if fname.startswith(prefix)]
        matched_files.extend(matched)
    return matched_files
