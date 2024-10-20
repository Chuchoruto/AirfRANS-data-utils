import pandas as pd
import numpy as np
import os
import joblib

def convert_and_filter_dataframes(
    dataset_list, 
    desired_columns=[0, 1, 5, 8, 9], 
    column_names=['x', 'y', 'sdf', 'v_x', 'v_y']
):
    """
    Converts a list of NumPy arrays or similar structures into DataFrames,
    keeping only the desired columns and assigning custom column names.

    Parameters:
        dataset_list (list): A list where each element is a NumPy array or 2D numerical data.
        desired_columns (list): List of column indices to keep. Default is [0, 1, 5, 8, 9].
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
                print(f"Converted and filtered dataset {i + 1} with shape {filtered_df.shape}.")
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
            print(f"Saved DataFrame {i + 1} to {file_path}.")
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

        print(f"Loaded batch {i // batch_size + 1} with {len(batch)} DataFrames.")
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

