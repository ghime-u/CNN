# Convolutional Neural Network Project Readme

## Overview

This repository contains the code and documentation for a Convolutional Neural Network (CNN) project that focuses on classifying datasets containing digits and Greek symbols. The project involves training a CNN on the MNIST dataset, examining the network's filters, creating a digit embedding space, and conducting experiments to analyze the impact of different parameters.

## Project Structure

The project is organized into several tasks, each addressing a specific aspect of CNNs and image classification.

### Task 1: Build and Train a Network to Recognize Digits

#### Part A: Get the MNIST Dataset
- Utilized the TorchVision library to import the MNIST dataset.
- Created loaders for training and testing datasets with specified batch sizes.
- Displayed the first 6 digits using Matplotlib.

#### Part B: Make Your Network Code Repeatable
- Set a manual seed using `torch.manual_seed(69)` for reproducibility.
- Disabled CUDA to ensure code repeatability.

#### Part C: Build a Network Model
- Created a CNN model with the following architecture:
  - Input -> Conv2D (10 channels, 5x5 kernel) -> Max Pool (2x2) -> ReLU -> Conv2D (20 filters, 10 input channels, 5x5 kernel) -> Dropout (0.5) -> Max Pooling -> ReLU -> Flatten -> Linear Transformation (320 -> 50) -> Linear Transformation (50 -> 10).

#### Part D: Train the Model
- Trained the model for 5 epochs with a batch size of 640 for training and 2000 for testing.

#### Part E: Save the Network to a File
- Saved the model and optimizer files using `network.state_dict()` on Google Drive for easy access.

#### Part F: Read the Network and Run it on Test Data
- Loaded the saved model using `load_state_dict(torch.load)`.
- Evaluated the model on the MNIST test dataset.

#### Part G: Test the Network on New Inputs
- Uploaded a folder of handwritten digits to Google Drive.
- Used `torchvision.datasets.ImageFolder` to create a testing dataset.
- Tested the trained network on new images, achieving 4/10 recognition accuracy.

### Task 2: Examine Your Network

#### Part A: Analyze the First Layer
- Extracted and printed the weights of the first layer, revealing filters detecting edges and shapes, including Sobel filters for intensity.

#### Part B: Show the Effect of Filters
- Applied 20 2D filters using OpenCV functions with `torch.no_grad()`.

#### Part C: Build a Truncated Model
- Created a truncated model with only the first two convolutional layers.
- Applied the truncated model to the first image, displaying 10 4x4 filters and their impact on the input image.

### Task 3: Create a Digit Embedding Space

#### Part A: Create a Greek Symbol Dataset
- Generated a dataset of Greek symbols using `torchvision.dataset.ImageFolder`.
- Transformed and processed the data, saved flattened images to a CSV file, and stored image paths and labels in another CSV file.

#### Part B: Create a Truncated Model: Tensor of 50 Numbers as Output
- Created a truncated model with an output tensor of 50 numbers.

#### Part C: Project the Greek Symbol into the Truncated Space
- Applied the model to the original CSV file, obtaining 27 sets of 50 numbers.

#### Part D: Compute the SSD Values
- Computed the Sum of Squared Differences (SSD) values for all pairs of 27 images.
- Observed close distances for images of the same type, confirming effective categorization.

#### Part E: Create Your Own Symbol
- Generated new Greek symbols and validated their similarity using SSD values.

### Task 4: Design Your Own Experiment

#### Part A: Metrics Used
- Explored the impact of the number of epochs and batch size on the model's performance.

#### Part B and C: Predict and Evaluate the Result
- Analyzed loss tables for different numbers of epochs and batch sizes.
- Explored the effect of adding a dropout layer after the second fully connected layer.

## Conclusion

This project provides a comprehensive exploration of CNN architecture, filter analysis, digit embedding spaces, and experimentation with different parameters. The README.md serves as a guide to understanding the structure of the project and its key findings.

## Acknowledgements

- PyTorch Tutorials: [https://pytorch.org/tutorials/beginner/basics/intro.html](https://pytorch.org/tutorials/beginner/basics/intro.html)
- PyTorch Documentation: [https://pytorch.org/](https://pytorch.org/)
- Stack Overflow: Various contributions for creating custom datasets.
