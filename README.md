🍬 Sugar Toxicity Classification Model

This project trains a neural network to decide whether a mixture of sugar concentrations and exposure time becomes toxic or remains safe.

It’s built with PyTorch and uses synthetic data to simulate different scenarios, making it ideal for experimentation, learning, and prototyping.

🧠 What This Project Does

Given:

Four sugar concentration values

An exposure duration (in hours)

The model predicts:

Non-toxic (0)

Toxic (1)

The focus is on building a clean, modular machine learning pipeline rather than solving a real-world medical or chemical problem.

🧪 Synthetic Data

All data is artificially generated to cover a wide range of scenarios.

Input Features

sugar_1

sugar_2

sugar_3

sugar_4

hours

Toxicity Logic

A continuous toxicity score is calculated using a weighted combination of the sugar values and exposure time.
This score is normalized and converted into a binary label:

Toxic if the score is greater than 0.65

Non-toxic otherwise

Both the raw toxicity score and the final label are stored in the dataset.

📦 Dataset Handling

The dataset is wrapped in a custom PyTorch Dataset class that:

Loads the CSV file

Normalizes all numeric inputs to the range [0, 1]

Returns tensors ready for training

Provides integer class labels for classification

This keeps preprocessing clean and reusable.

🧱 Model Architecture

The classifier is a fully connected neural network.

Structure Overview

Input layer with 5 features

Three hidden layers with ReLU activations

Output layer with 2 neurons (binary classification)

The model outputs logits, which are used directly with cross-entropy loss.

🚀 Training Workflow

The training pipeline includes:

Dataset split into training, validation, and test sets

Batch-based training with PyTorch data loaders

AdamW optimizer

Learning rate scheduling

Validation loss monitoring after each epoch

Model checkpoint saving

GPU acceleration is supported if available.

📊 Evaluation

Validation loss is computed at the end of each epoch

Final testing is performed on unseen data

Test loss is reported to measure generalization

The evaluation setup is intentionally simple and easy to extend.

🎯 Purpose & Scope

This project is designed for:

Learning PyTorch fundamentals

Practicing dataset and model design

Experimenting with classification problems

Building an end-to-end ML workflow

It uses synthetic data and should not be treated as a real toxicity prediction system.
