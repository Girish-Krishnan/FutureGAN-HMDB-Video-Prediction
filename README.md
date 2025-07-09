# FutureGAN Video Prediction on HMDB dataset

## Description

This repository contains the implementation of FutureGAN, a Generative Adversarial Network for video prediction. FutureGAN is able to generate future frames of a video given a sequence of previous frames. This project can be beneficial for multiple applications such as video prediction for surveillance videos, autonomous vehicles, weather forecasting, etc.

## Table of Contents

- [FutureGAN Video Prediction on HMDB dataset](#futuregan-video-prediction-on-hmdb-dataset)
  - [Description](#description)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Getting the Dataset](#getting-the-dataset)
  - [Usage](#usage)
  - [Project Structure](#project-structure)

## Installation

To run this project, you will need Python 3.x and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [PyTorch](https://pytorch.org/)
- [Matplotlib](https://matplotlib.org/)
- [OpenCV](https://opencv.org/)

To install the project, do the following:

1. Clone the repo
    ```
    git clone https://github.com/Girish-Krishnan/Video_Prediction.git
    ```
2. Install the requirements
    ```
    pip install -r requirements.txt
    ```

## Getting the Dataset

The dataset can be found as a `.zip` file on this [Google Drive link](https://drive.google.com/file/d/1yPMWhr_-4YZenPI_HRNoGVQGDsENKwKb/view). Extract it to the `./data` folder.

## Usage

To train the model:

```
python train.py --data_root <path_to_data_directory> --n_epochs <number_of_epochs>
```

To evaluate the model and save metrics:

```
python evaluate.py --model_path <path_to_model_checkpoint> --data_root <path_to_data_directory>
```

To plot the saved metrics:

```
python plot_eval_metrics.py --metrics_path <path_to_saved_metrics>
```

To generate a video prediction:

```
python predict_video.py --model_path <path_to_model_checkpoint> --data_root <path_to_data_directory> --output_path <path_for_generated_video>
```


## Project Structure

The repository has the following structure:

- `models.py` - Contains the PyTorch implementation of the Generator and Discriminator for FutureGAN.
- `dataset.py` - Contains the custom PyTorch dataset for video data.
- `train.py` - Trains the FutureGAN model using the specified training data.
- `evaluate.py` - Evaluates the model and saves the evaluation metrics (MSE and PSNR).
- `plot_eval_metrics.py` - Plots the evaluation metrics over epochs.
- `predict_video.py` - Uses the trained model to predict future frames of a video and creates a video of the predictions.
- `requirements.txt` - Contains all the python packages required to run the project.
- `config.yaml` - Contains all the hyperparameters for training the FutureGAN.
