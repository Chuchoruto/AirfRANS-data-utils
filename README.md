# AirfRANS-data-utils

Welcome to the **AirfRANS Data Utils** repository! This repository provides tools to manipulate, visualize, and package the AirfRANS dataset for machine learning applications.

## Table of Contents
- [Introduction](#introduction)
- [Repository Structure](#repository-structure)
- [Design Decisions](#design-decisions)
- [Visualization Using Heatmaps](#visualization-using-heatmaps)
- [How to Use](#how-to-use)


## Introduction
The AirfRANS dataset contains numerical solutions of the **Reynolds-Averaged Navier–Stokes (RANS) equations**. These equations describe fluid dynamics over a variety of airfoils in subsonic flight conditions, providing useful data for both CFD research and machine learning experiments.

## Repository Structure

/data_manip/
    - Code for data manipulation and visualization
/modeling/
    - AirFNN class used to train the model
/heatmaps/
    - Heatmap visualizations of two simulations 
/data_distribution_plots/
    - Plots of data to give understanding of distribution
usage.ipynb
    - Shows functionality of code and creates all visualizations

## Design Decisions

- Converted the simulations to pandas dataframes and filtered out non desired columns
- Packaged data as Torch tensors for training
    - Experimented with dataloader, but it was running quite slow. Definitely are use cases where it would be useful.
- Used simple neural network with MSE loss just to show training capability


## Visualizations Using Heatmaps

The heatmaps help visualize the simulation and show metrics relative to the airfoil:
    - Implicit distance in the form of SDF is shown
    - Velocity in the x direction is shown
    - Velocity in the y direction is shown

In each of the velocity heatmaps you can see the outline of the airfoil in certain parts
The SDF plot shows radial distance away from the airfoil as expected

## How to Use

### Cloning repo

run git clone `https://github.com/Chuchoruto/AirfRANS-data-utils.git`

### Install dependencies:

Run `pip install requirements.txt` to make sure no dependencies are missing

### Using as standalone in Google Colab

You can run:
```
!wget https://raw.githubusercontent.com/Chuchoruto/AirfRANS-data-utils/main/data_manip/manip_utils.py
import manip_utils as manip
!wget https://raw.githubusercontent.com/Chuchoruto/AirfRANS-data-utils/main/modeling/AirFNN.py
import AirFNN
```
This will let you use all of the manipulations and utilities from the modules


