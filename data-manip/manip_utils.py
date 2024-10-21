import pandas as pd
import numpy as np
import os
import joblib
import torch
from torch.utils.data import TensorDataset, ConcatDataset
def convert_and_filter_dataframes(
    dataset_list, 
    desired_columns=[0, 1, 4, 8, 9], 
    column_names=['x', 'y', 'sdf', 'v_x', 'v_y']
):
    """
    Converts a list of NumPy arrays or similar structures into DataFrames,
    keeping only the desired columns and assigning custom column names.

    Parameters:
        dataset_list (list): A list where each element is a NumPy array or 2D numerical data.
        desired_columns (list): List of column indices to keep. Default is [0, 1, 4, 8, 9].
        column_names (list): List of names to assign to the filtered columns. 
                             Default is ['x', 'y', 'sdf', 'v_x', 'v_y'].

    Returns:
        list: A list of filtered and renamed Pandas DataFrames.
    """
    if len(desired_columns) != len(column_names):
        raise ValueError("The number of desired columns must match the number of column names.")

    dataframes = []  # Initialize an empty list to store DataFrames
    
    for i, dataset in enumerate(dataset_list):
        try:
            # Ensure the dataset is a NumPy array or convertible
            if isinstance(dataset, np.ndarray):
                df = pd.DataFrame(dataset)  # Convert NumPy array to DataFrame
                
                # Keep only the desired columns
                filtered_df = df.iloc[:, desired_columns]

                # Rename columns
                filtered_df.columns = column_names

                dataframes.append(filtered_df)
                
            else:
                raise ValueError(f"Dataset {i + 1} is not a valid NumPy array.")
        except Exception as e:
            print(f"Error converting dataset {i + 1}: {e}")
    
    print(f"Total DataFrames created: {len(dataframes)}")
    return dataframes



def save_dataframes_as_bytes(dataframes, directory='./processed_data/'):
    """
    Saves a list of DataFrames as binary files using joblib.

    Parameters:
        dataframes (list): List of Pandas DataFrames to be saved.
        directory (str): Directory to save the binary files. Default is './processed_data/'.
    """
    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    for i, df in enumerate(dataframes):
        try:
            # Save each DataFrame as a joblib file
            file_path = os.path.join(directory, f'df_{i}.joblib')
            joblib.dump(df, file_path)
            
        except Exception as e:
            print(f"Error saving DataFrame {i + 1}: {e}")

    print(f"Total DataFrames saved: {len(dataframes)}")



def load_dataframes_in_batches(directory='./processed_data/', batch_size=10):
    """
    Loads DataFrames from joblib files in batches.

    Parameters:
        directory (str): Directory containing the saved joblib files. Default is './processed_data/'.
        batch_size (int): Number of DataFrames to load per batch.

    Yields:
        list: A batch of loaded DataFrames.
    """
    # Get the list of all joblib files in the directory, sorted by filename
    files = sorted([f for f in os.listdir(directory) if f.endswith('.joblib')])
    
    if not files:
        print("No joblib files found in the directory.")
        return

    # Load DataFrames in batches
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i + batch_size]
        batch = []
        
        for f in batch_files:
            try:
                # Load each DataFrame and add to the batch
                df = joblib.load(os.path.join(directory, f))
                batch.append(df)
            except Exception as e:
                print(f"Error loading {f}: {e}")

        
        yield batch

def load_dataframes_in_batches_and_collect(directory='./processed_data/', batch_size=10):
    """
    Loads DataFrames from joblib files in batches and collects them into a single list.

    Parameters:
        directory (str): Directory containing the saved joblib files. Default is './processed_data/'.
        batch_size (int): Number of DataFrames to load per batch.

    Returns:
        list: A list of all loaded DataFrames.
    """
    all_dataframes = []  # Initialize an empty list to store all DataFrames

    for batch in load_dataframes_in_batches(directory, batch_size):
        all_dataframes.extend(batch)  # Add each batch to the main list

    print(f"Total DataFrames loaded: {len(all_dataframes)}")
    return all_dataframes





class ChunkedDataset(Dataset):
    def __init__(self, directory, chunk_prefix):
        self.directory = directory
        self.chunk_files = sorted([f for f in os.listdir(directory) if f.startswith(chunk_prefix)])
        self.chunk_sizes = []
        self.total_size = 0
        
        for file in self.chunk_files:
            size = os.path.getsize(os.path.join(directory, file))
            self.chunk_sizes.append(size // (4 * 5))  # Assuming float32 (4 bytes) and 5 columns
            self.total_size += self.chunk_sizes[-1]

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        for i, size in enumerate(self.chunk_sizes):
            if idx < size:
                chunk = torch.load(os.path.join(self.directory, self.chunk_files[i]))
                return chunk[idx]
            idx -= size
        raise IndexError("Index out of range")

def package_dataframes_for_training(dataframes, chunk_size=10000, output_dir='./chunked_data'):
    """
    Packages a list of DataFrames into chunks and saves them to disk.

    Parameters:
        dataframes (list): List of DataFrames, each containing 'x', 'y', 'v_x', 'v_y', and 'sdf' columns.
        chunk_size (int): Number of rows to process at once within each DataFrame.
        output_dir (str): Directory to save the chunked data.

    Returns:
        ChunkedDataset: A custom dataset that loads chunks from disk as needed.
    """
    os.makedirs(output_dir, exist_ok=True)
    chunk_count = 0
    total_samples = 0

    for i, df in enumerate(dataframes):
        for start in range(0, len(df), chunk_size):
            end = start + chunk_size
            chunk = df.iloc[start:end]
            
            X_data = chunk[['x', 'y']].values
            Y_data = chunk[['v_x', 'v_y', 'sdf']].values

            combined_data = torch.tensor(np.hstack((X_data, Y_data)), dtype=torch.float32)

            torch.save(combined_data, os.path.join(output_dir, f'chunk_{chunk_count}.pt'))
            chunk_count += 1
            total_samples += len(X_data)

        print(f"Processed DataFrame {i+1}/{len(dataframes)}, Total samples: {total_samples}")
        del df  # Free up memory
        
    print(f"Packaged data into {chunk_count} chunks with {total_samples} total samples.")
    return ChunkedDataset(output_dir, 'chunk_')
