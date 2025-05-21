# utils/preprocessor.py

import numpy as np
import os
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
from tqdm import tqdm


def get_x_y_data(file_list, chroma_dir, metadata):
    """
    Returns x (features) and y (labels) for a given list of audio files.
    
    Args:
        file_list (list): List of audio filenames to load.
        chroma_dir (str): Directory path where chromagram .npy files are stored.
        metadata (pd.DataFrame): Metadata DataFrame containing raga labels and filenames.

    Returns:
        x_data (np.ndarray): Feature array loaded using get_x_data.
        y_labels (np.ndarray): Label array loaded using get_y_labels.
    """
    all_file_names=os.listdir(chroma_dir)
    files = match_filenames_by_prefix(all_file_names, file_list)
    # print(len(files))
    x_data = get_x_data(files, chroma_dir,metadata)
    print("X_data Done")
    # print(x_data.shape)
    y_labels, encoder = get_y_labels(files,chroma_dir, metadata)
    return x_data, y_labels, encoder.classes_


def get_x_data(filenames, chroma_dir, df,chroma_bins = 320,num_frames= 938):
    """
    Prepares chromagram input data for model training.

    Parameters:
    filenames (list): List of available chromagram file names
    chroma_dir (str): Directory containing chromagram .npy files
    df (pd.DataFrame): Metadata with tonic and shift info
    split_names (list): List of file names belonging to the split

    Returns:
    np.ndarray: Array of processed chromagrams (num_samples, num_frames, chroma_bins, 1)
    """

    x_data = []
    df=get_shift_value(df)

    for fname in tqdm(filenames, desc="Processing files"):
        file_path = os.path.join(chroma_dir, fname)
        chroma = np.load(file_path)

        # Validate shape
        if chroma.shape != (chroma_bins, num_frames):
            continue

        # Transpose and reshape
        chroma = chroma.T  # shape becomes (938, 12)
        chroma = chroma.reshape(num_frames, chroma_bins, 1)

        # Get tonic shift
        matched_row = get_matched_row(df, fname)
        if matched_row is None:
            base_name = "_".join(fname.split("_")[:-1])
            print(f"Warning: No shift value found for base name: {base_name}")
            continue
        shift_value = int(matched_row["shift"])

        # Shift chromagram
        shifted_chroma = shift_columns(chroma, shift_value)
        x_data.append(shifted_chroma)

    return np.array(x_data)



import os
import numpy as np
from sklearn.preprocessing import LabelBinarizer

def get_y_labels(filenames, chroma_dir, df, chroma_bins=320, num_frames=938, fit_encoder=False, encoder=None):
    """
    Extracts the corresponding raga labels for a given split and returns one-hot encoded labels.

    Parameters:
    filenames (list): List of filenames to fetch labels for
    chroma_dir (str): Path to chroma features
    df (pd.DataFrame): Metadata with 'audio_file' and 'raga name' columns
    chroma_bins (int): Expected number of chroma bins
    num_frames (int): Expected number of time frames in chroma
    fit_encoder (bool): If True, creates and fits a new encoder
    encoder (LabelBinarizer): Existing encoder to use (e.g., for val/test splits)

    Returns:
    tuple: (np.ndarray of one-hot encoded labels, LabelBinarizer)
    """
    y_labels = []

    for fname in filenames:
        file_path = os.path.join(chroma_dir, fname)
        chroma = np.load(file_path)
        # print(chroma.shape)
        if chroma.shape != (chroma_bins, num_frames):
            print("Incorrect Shape, file skipped")
            continue
        matched_row = get_matched_row(df, fname)
        if matched_row is None:
            continue
        raga = matched_row["raga name"]
        # raga = df[df["audio_file"] == fname]["raga name"].values[0]
        y_labels.append(raga)

    y_labels = np.array(y_labels)
    y_onehot, encoder = one_hot_encode_labels(y_labels, fit_encoder, encoder)

    return y_onehot, encoder


def match_filenames_by_prefix(all_file_names, file_list):
    matched = [fname for fname in all_file_names if any(fname.startswith(prefix) for prefix in file_list)]
    return matched


def get_matched_row(df, fname):
    """
    Given a filename, extract base name (by removing the last underscore segment) 
    and return the matching row from df where 'audio_file' equals the base name.
    """
    # Strip file extension and extract base name
    base_name = "_".join(os.path.splitext(fname)[0].split("_")[:-1])
    
    # Find the matching row
    matched_row = df[df["audio_file"] == base_name]
    
    if matched_row.empty:
        print(f"Warning: No match found for base name: {base_name}")
        return None
    return matched_row.iloc[0] 


def one_hot_encode_labels(y_labels, fit_encoder=False, encoder=None):
    if fit_encoder or encoder is None:
        encoder = LabelBinarizer()
        y_onehot = encoder.fit_transform(y_labels)
    else:
        y_onehot = encoder.transform(y_labels)

    return y_onehot, encoder


def get_shift_value(df):
    # Note mapping with "G" as the reference
    note_mapping = {
        "G": 0,
        "G#": 1,
        "A": 2,
        "A#": 3,
        "B": 4,
        "C": 5,
        "C#": 6,
        "D": 7,
        "D#": 8,
        "E": 9,
        "F": 10,
        "F#": 11
    }

    # Identify unmapped tonic values
    unique_tonics = df["tonic"].unique()
    unmapped_tonics = [tonic for tonic in unique_tonics if tonic not in note_mapping]

    if unmapped_tonics:
        print("Unmapped tonic values found:", unmapped_tonics)

    # Convert notes to numbers
    df["shift"] = df["tonic"].map(note_mapping)
    return df



def shift_columns(arr, N):
    shifted_arr = np.roll(arr, -N, axis=1)
    return shifted_arr
