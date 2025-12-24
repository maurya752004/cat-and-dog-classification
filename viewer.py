#!/usr/bin/env python3
"""
Simple image viewer: shows the image on the left and a side box
with the label (cat/dog) inferred from the image's parent folder.

Usage:
  python viewer.py --path dataset/data/train/cat/cat.1.jpg
  python viewer.py            # picks a random image from dataset/data/test
"""
import argparse
import os
import random
import sys
import time
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
try:
    import tensorflow as tf
except Exception:
    tf = None

# Model image size (matches `main.py`)
IMG_SIZE = 150


def find_random_image(root='dataset/data'):
    # Search for jpg/png images under root
    candidates = []
    for sub, dirs, files in os.walk(root):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                candidates.append(os.path.join(sub, f))
    return random.choice(candidates) if candidates else None


def get_label_from_path(path):
    # Parent folder name is expected to be 'cat' or 'dog'
    parent = os.path.basename(os.path.dirname(path))
    return parent.capitalize()


def show_image_with_label(path, predicted=None, prob=None):
    img = Image.open(path).convert('RGB')
    true_label = get_label_from_path(path)

    fig = plt.figure(figsize=(8, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.05)

    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(img)
    ax_img.axis('off')

    ax_box = fig.add_subplot(gs[0, 1])
    ax_box.axis('off')
    rect = plt.Rectangle((0.05, 0.12), 0.9, 0.76, facecolor='#f8f8f8', edgecolor='k')
    ax_box.add_patch(rect)

    # Build label text: predicted (with probability) and true label
    if predicted is not None and prob is not None:
        label_lines = f"Pred: {predicted}\n{prob*100:.1f}%\nTrue: {true_label}"
    elif predicted is not None:
        label_lines = f"Pred: {predicted}\nTrue: {true_label}"
    else:
        label_lines = f"True: {true_label}"

    ax_box.text(0.5, 0.5, label_lines, fontsize=18, ha='center', va='center')

    fig.suptitle(os.path.basename(path), fontsize=10)
    return fig


def predict_label(path, model):
    # Preprocess image for model: resize, scale, batch
    img = Image.open(path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype('float32') / 255.0
    arr = np.expand_dims(arr, 0)
    preds = model.predict(arr)
    # Expect binary output (sigmoid)
    prob = float(preds[0][0]) if preds.shape[-1] == 1 or preds.ndim == 1 else float(preds[0][1])
    label = 'Dog' if prob >= 0.5 else 'Cat'
    return label, prob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', help='Path to image file')
    parser.add_argument('--root', '-r', default='dataset/data', help='Dataset root to pick random image')
    parser.add_argument('--save', action='store_true', help='Save viewer output to a file instead of showing')
    parser.add_argument('--output', '-o', help='Output filename when saving (e.g. out.png)')
    parser.add_argument('--model', '-m', default='model/cat_dog_cnn.h5', help='Path to saved Keras model for prediction')
    parser.add_argument('--no-predict', action='store_true', help="Don't run model prediction even if model exists")
    args = parser.parse_args()

    img_path = args.path or find_random_image(args.root)
    if not img_path:
        print('No images found under', args.root)
        return

    if not os.path.exists(img_path):
        print('Image path does not exist:', img_path)
        return

    # Attempt to load model (if requested and available)
    model = None
    if not args.no_predict and args.model and os.path.exists(args.model) and tf is not None:
        try:
            model = tf.keras.models.load_model(args.model)
            print('Loaded model from', args.model)
        except Exception as e:
            print('Failed to load model:', e)

    predicted = None
    prob = None
    if model is not None:
        try:
            predicted, prob = predict_label(img_path, model)
        except Exception as e:
            print('Prediction failed:', e)

    fig = show_image_with_label(img_path, predicted=predicted, prob=prob)

    # Decide whether to show interactively or save to file.
    need_save = args.save or ('DISPLAY' not in os.environ and sys.platform != 'win32')
    if args.output:
        outpath = args.output
    else:
        outpath = f"viewer_output_{int(time.time())}.png"

    if need_save:
        fig.savefig(outpath, bbox_inches='tight')
        print('Saved viewer image to', outpath)
    else:
        plt.show()


if __name__ == '__main__':
    main()
