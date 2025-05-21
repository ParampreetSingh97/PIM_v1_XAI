"""
Main script for training and evaluating the CNN model for Raga classification.
"""

from utils.preprocess import load_required_filenames, load_metadata,create_train_val_test_splits
from utils.data_loader import get_x_y_data
from utils.model import make_model
from utils.trainer import train_model, evaluate_model

def main():
    # Path to chromagram files, metadata and list containing required Raga names
    chroma_dir = "/hdd_storage/data/PB/param/spectogram_501_renamed_upload/"
    metadata_path = "/home/parampreet/Raga_model_chroma_modified/data/Metadata.csv"
    required_ragas=['Bhairavi', 'Bihag', 'Des', 'Jog', 'Kedar', 'Khamaj', 'Malkauns',
        'Maru_bihaag', 'Nayaki_kanada', 'Shuddha_kalyan', 'Sohni', 'Yaman']

    # Load filenames and metadata
    metadata = load_metadata(metadata_path)
    filenames, metadata = load_required_filenames(metadata, required_ragas)

    # Split data into train, val, and test sets
    train_files,val_files,test_files=create_train_val_test_splits(metadata, chroma_dir)

    ##load chromagrams and labels
    print("Getting Train Files!")
    x_train,y_train,class_names=get_x_y_data(train_files, chroma_dir, metadata)
    print("Getting Validation Files!")
    x_val,y_val,_=get_x_y_data(val_files, chroma_dir, metadata)
    print("Getting Test Files!")
    x_test,y_test,_=get_x_y_data(test_files, chroma_dir, metadata)

    # Build model
    model = make_model(input_shape=x_train.shape[1:], num_classes=y_train.shape[1])

    # Print Summary of the model
    print(model.summary())

    # Train model
    model, history=train_model(model, x_train, y_train, x_val, y_val, epochs=10, batch_size=32)

    # Evaluate model
    report=evaluate_model(model, x_test, y_test, class_names)

    # Save model
    model.save("final_model.keras")

if __name__ == "__main__":
    main()
