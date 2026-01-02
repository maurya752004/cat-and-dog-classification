# cat-and-dog-classification

A small TensorFlow/Keras project that trains a convolutional neural network (CNN) to classify images of cats and dogs. This repository includes training and simple viewing utilities, a requirements file, and a sample output image.

Contents
- README.md — usage, setup, architecture, examples, and contribution guidelines
- main.py — training script (CLI)
- viewer.py — lightweight image viewer + optional prediction helper
- requirements.txt — minimal Python packages required
- .gitignore — ignored files (including model weights)
- dataset/ — expected dataset layout (not included in repo)
- model/ — model output (created by training)
- out.png — example image (preview)

Table of contents
1. Project Overview
2. Features
3. Dataset layout & preparation
4. Requirements
5. Quick start (set up and run)
6. Training (examples & options)
7. Model architecture
8. Evaluation & checkpoints
9. Viewer utility
10. Results / example output
11. Tips for improving accuracy
12. Reproducibility & environment
13. Contributing
14. License
15. Acknowledgements & references
16. Contact

1. Project overview
This repo provides code to train a binary image classifier (Cat vs Dog) using a small CNN built with TensorFlow/Keras. It uses ImageDataGenerator for augmentation and supports saving the final model and best checkpoint.

2. Features
- Data augmentation for robust training (rotation, shifts, shear, zoom, horizontal flip).
- Model with BatchNormalization, Dropout and three Conv2D blocks.
- EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint callbacks.
- CLI flags for epochs, batch size, dataset path and test-run.
- Viewer utility to display images and predicted/true labels.

3. Dataset layout & preparation
Expected dataset directory structure (default: dataset/data/train):

- dataset/
  - data/
    - train/
      - cat/
        - cat.1.jpg
        - ...
      - dog/
        - dog.1.jpg
        - ...
    - test/ (optional)
      - cat/
      - dog/

Notes:
- main.py expects --train-dir (default dataset/data/train).
- ImageDataGenerator uses validation_split=0.2, so training and validation are split from the same folder.

4. Requirements
- Python 3.8+
- See requirements.txt (tensorflow>=2.10, numpy)

Install dependencies:

python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\Activate

pip install -r requirements.txt

5. Quick start (set up and run)
1. Place dataset into dataset/data following the structure above.
2. Create and activate a virtual environment and install requirements.
3. Run a short test training (1 epoch) to verify everything runs:

python main.py --test-run

4. For full training:

python main.py --epochs 25 --batch-size 32 --train-dir dataset/data/train --model-out model/cat_dog_cnn.h5

6. Training (examples & CLI)
Available CLI options in main.py:
- --epochs         : number of epochs (default 25)
- --batch-size     : batch size (default 32)
- --train-dir      : training directory (default dataset/data/train)
- --model-out      : output path for final model (default model/cat_dog_cnn.h5)
- --test-run       : run a short test (sets epochs to 1)

Callbacks used:
- EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
- ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
- ModelCheckpoint(... '_best.h5', save_best_only=True)

Notes on steps:
- steps_per_epoch = max(1, train_data.samples // batch_size)
- validation_steps = max(1, val_data.samples // batch_size)

7. Model architecture (from main.py)
- Input: (150, 150, 3)
- Conv2D(32, 3x3) -> BatchNormalization -> MaxPooling2D(2x2)
- Conv2D(64, 3x3) -> BatchNormalization -> MaxPooling2D(2x2)
- Conv2D(128, 3x3) -> BatchNormalization -> MaxPooling2D(2x2)
- Flatten
- Dense(256, activation='relu')
- Dropout(0.5)
- Dense(1, activation='sigmoid')

Optimizer: Adam(lr=1e-4), Loss: binary_crossentropy, Metrics: accuracy

8. Evaluation & checkpoints
- Best model checkpoint saved as model/cat_dog_cnn_best.h5 (ModelCheckpoint pattern).
- Final model saved to path specified by --model-out.
- Use model.evaluate or model.predict for evaluation on a held-out test set.

9. Viewer utility (viewer.py)
- Purpose: display an image and a side box showing the true label (inferred from parent directory) and predicted label/probability if a TensorFlow model is available.
- Usage:
  - View a specific image:

python viewer.py --path dataset/data/train/cat/cat.1.jpg

  - Pick a random image (default root dataset/data):

python viewer.py

- Prediction: viewer.py resizes image to 150x150 and expects a model that outputs a single sigmoid probability. Probability >= 0.5 => Dog, else Cat.

10. Results / example output
- out.png is included as a sample/preview image.
- After training, check model/ for saved .h5 files.

11. Tips for improving accuracy
- Use transfer learning (MobileNetV2, ResNet50) and fine-tune.
- Increase dataset size or use external datasets.
- Apply more varied augmentation.
- Use class weights or focal loss for imbalance.
- Try different learning rates and optimizers or learning rate schedules.

12. Reproducibility & environment
- Pin package versions in requirements.txt for reproducibility.
- To set seeds for repeatability:

import random
import numpy as np
import tensorflow as tf

random.seed(123)
np.random.seed(123)
tf.random.set_seed(123)

13. Contributing
- Fork the repo, create a branch, make changes, and open a pull request.
- Suggested improvements: add evaluation scripts, unit tests, Dockerfile, CI workflows.

14. License
This project is licensed under the MIT License — see the LICENSE file for details.

15. Acknowledgements & references
- Built with TensorFlow and Keras.
- ImageDataGenerator usage inspired by Keras documentation and common tutorials.
- Consider using the Kaggle Dogs vs. Cats dataset for experimentation (ensure you comply with dataset terms).

16. Contact
Repository owner: @maurya752004
