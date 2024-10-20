import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
def plot_simulation_heatmaps(simulation_df, output_dir='./heatmaps/', simulation_index=0):
    """
    Creates heatmaps for v_x, v_y, and sdf for a given simulation DataFrame.

    Parameters:
        simulation_df (pd.DataFrame): DataFrame with x, y, v_x, v_y, and sdf columns.
        output_dir (str): Directory to save the heatmaps. Default is './heatmaps/'.
        simulation_index (int): Index of the simulation for naming purposes.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Reduce data points by sampling
    sample_size = min(100000, len(simulation_df))
    simulation_df = simulation_df.sample(n=sample_size)

    # Create grids more efficiently
    x = np.linspace(simulation_df['x'].min(), simulation_df['x'].max(), 100)
    y = np.linspace(simulation_df['y'].min(), simulation_df['y'].max(), 100)
    X, Y = np.meshgrid(x, y)

    for value in ['v_x', 'v_y', 'sdf']:
        plt.figure(figsize=(8, 6))
        plt.tricontourf(simulation_df['x'], simulation_df['y'], simulation_df[value], levels=20, cmap='coolwarm')
        plt.colorbar()
        plt.title(f'Simulation {simulation_index} - {value} Heatmap')
        plt.savefig(os.path.join(output_dir, f'simulation_{simulation_index}_{value}.png'), dpi=100)
        plt.close()

    print(f"Heatmaps for simulation {simulation_index} saved to {output_dir}.")

def generate_heatmaps_for_simulations(dataframes, selected_indices, output_dir='./heatmaps/'):
    """
    Generates heatmaps for a selected list of simulations.

    Parameters:
        dataframes (list): List of DataFrames for all simulations.
        selected_indices (list): List of indices specifying which simulations to plot.
        output_dir (str): Directory to save the heatmaps. Default is './heatmaps/'.
    """
    for idx in selected_indices:
        try:
            simulation_df = dataframes[idx]
            plot_simulation_heatmaps(simulation_df, output_dir=output_dir, simulation_index=idx)
        except IndexError:
            print(f"Simulation {idx} not found in the provided dataframes.")


def print_simulation_statistics(df):
    """
    Prints the min, max, mean, and standard deviation for each column in the given DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing the columns 'x', 'y', 'v_x', 'v_y', and 'sdf'.
    """
    # Compute min, max, mean, and std for each column
    stats = df.describe().loc[['min', 'max', 'mean', 'std']]
    print(stats)

def plot_data_distributions(df, simulation_index=0, output_dir='./data_distribution_plots'):
    """
    Plots the distribution of x, y, v_x, v_y, and sdf and saves them to a specified directory.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'x', 'y', 'v_x', 'v_y', and 'sdf' columns.
        simulation_index (int): Index of the simulation for naming the plots.
        output_dir (str): Directory to save the distribution plots. Default is './data_distribution_plots'.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Plot the distribution for each column
    for column in ['x', 'y', 'v_x', 'v_y', 'sdf']:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[column], kde=True, bins=100)
        plt.title(f'Distribution of {column} - Simulation {simulation_index}')
        plt.xlabel(column)
        plt.ylabel('Frequency')

        # Save the plot
        plot_path = os.path.join(output_dir, f'simulation_{simulation_index}_{column}_distribution.png')
        plt.savefig(plot_path)
        plt.close()

        print(f"Saved {column} distribution plot to {plot_path}.")