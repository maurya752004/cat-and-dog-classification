# Cat and Dog Classification

This repository contains a simple Convolutional Neural Network (CNN) project for binary image classification (cats vs dogs). It includes dataset layout, training code, a small image viewer utility, example saved models, and notes on handling large files.

## Contents

- `main.py` — training and model code (entry point for training)
- `viewer.py` — small image viewer and optional model prediction utility
- `requirements.txt` — Python dependencies
- `dataset/` — dataset folders (train, validation, test). See structure below.
- `model/` — saved model files (`cat_dog_cnn.h5`, `cat_dog_cnn_best.h5`). Note: large model files are excluded by `.gitignore` by default.

> IMPORTANT: The repository usually excludes large binary files such as virtual environments and model weights. See the "Model files and large assets" section below for how to manage pretrained models.

## Dataset structure

The expected layout under `dataset/data` is:

- `dataset/data/train/cat/`  (training cat images)
- `dataset/data/train/dog/`  (training dog images)
- `dataset/data/validation/cat/` (validation cat images)
- `dataset/data/validation/dog/` (validation dog images)
- `dataset/data/test/` (test images)

Files are expected to be common image formats: `.jpg`, `.jpeg`, `.png`.

If you don't have a dataset, you can prepare a small sample by placing images in the folders above or use public datasets such as the Kaggle Dogs vs Cats dataset.

## Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

If you don't have `requirements.txt` or want to install core packages manually, typical requirements include `tensorflow`, `numpy`, `Pillow`, and `matplotlib`.

Example minimal `requirements.txt` (use appropriate TF version for your hardware):

```
tensorflow>=2.6
numpy
Pillow
matplotlib
```

## Training

Run the training script (if present) with:

```bash
python main.py
```

Typical training considerations (check `main.py` for the concrete implementation):

- Image preprocessing: resize to a fixed `IMG_SIZE` (150 by default), rescale pixel values to `[0, 1]`.
- Architecture: a small CNN with several Conv2D + MaxPool blocks, followed by Dense layers and a sigmoid output for binary classification.
- Loss: `binary_crossentropy`.
- Optimizer: `adam` (typical), learning rate adjustments optional.
- Metrics: accuracy and validation loss/accuracy.

Adjust batch size, number of epochs and learning rate in `main.py` as needed for your machine.

## Viewer and quick inference

`viewer.py` shows an image and — if a Keras model is available — runs a prediction. Examples:

Show a random image from the test folder:

```bash
python viewer.py
```

Show a specific image and save the result to a file:

```bash
python viewer.py --path dataset/data/test/cat/cat.1.jpg --save --output out.png
```

Use a custom model for prediction (defaults to `model/cat_dog_cnn.h5`):

```bash
python viewer.py --model model/cat_dog_cnn.h5
```

Skip prediction even if a model exists:

```bash
python viewer.py --no-predict
```

## Model files and large assets

The `model/` directory may contain saved Keras models in HDF5 format (`.h5`). Large binary files (for example model weights or a full `venv/`) are excluded from the repository by `.gitignore` because GitHub enforces file size limits and it's not recommended to commit virtual environments.

If you need to add large models to the repo, use Git LFS (Large File Storage): https://git-lfs.github.com

Example (one-time setup):

```bash
# install git-lfs (system package manager or from https://git-lfs.github.com)
git lfs install
git lfs track "model/*.h5"
git add .gitattributes
git add model/your_model.h5
git commit -m "Add model via LFS"
git push origin main
```

Alternatively, keep model files in a cloud storage bucket and provide a download script.

### Removing accidentally committed large files

If you accidentally pushed large files and GitHub rejected or warned about them, remove them from history before pushing. Example using `git filter-repo` (recommended) or `git filter-branch` if `filter-repo` is not available:

```bash
# Example using git filter-repo (install separately)
git filter-repo --path venv/ --invert-paths
git filter-repo --path model/cat_dog_cnn.h5 --invert-paths

# Or with git filter-branch (slower, legacy):
git filter-branch --force --index-filter "git rm -r --cached --ignore-unmatch venv model/*.h5" --prune-empty --tag-name-filter cat -- --all

# Then garbage-collect and force-push cleaned history
rm -rf .git/refs/original/
git reflog expire --expire=now --all
git gc --prune=now --aggressive
git push origin --force --all
```

## Notes about pushing to GitHub


- The repository's `.gitignore` currently excludes `venv/` and `model/*.h5` to avoid committing large files.
- If you accidentally pushed large files and your push was rejected by GitHub, follow the instructions in the GitHub error message and use `git filter-repo`, `git filter-branch` or the `git lfs migrate` tools to remove them from history.

## Contributing

Feel free to open issues or pull requests. Keep model weights out of PRs — provide instructions to reproduce or download weights instead.

## License

Add your preferred license here (for example MIT) or create a `LICENSE` file in the repository.

## Contact

If you want help with training, model conversion, or publishing pretrained models, open an issue or contact the repository owner.

---

If you'd like, I can also:

- Add a `download_model.sh` script that downloads model weights from a cloud URL and places them in `model/`.
- Add example training logs or a short `RESULTS.md` summarizing observed validation accuracy after training.

Tell me which you'd like next.

---

If you want, I can:

- Add a small `download_model.sh` script to fetch pretrained weights from a cloud location.
- Add `README` badges or a short demo GIF showing `viewer.py` in action.

Tell me which you'd like next.
# cat-and-dog-classification
