import tensorflow as tf
import os
import argparse
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import callbacks

# Image parameters
IMG_SIZE = 150
BATCH_SIZE = 32
DEFAULT_EPOCHS = 25


def build_datagens():
    # Stronger augmentation for training, only rescaling for validation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2,
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.2,
    )

    return train_datagen, val_datagen


def build_model(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--train-dir', default='dataset/data/train')
    parser.add_argument('--model-out', default='model/cat_dog_cnn.h5')
    parser.add_argument('--test-run', action='store_true', help='Run a short test (1 epoch)')
    args = parser.parse_args()

    epochs = 1 if args.test_run else args.epochs
    batch_size = args.batch_size

    train_datagen, val_datagen = build_datagens()

    train_data = train_datagen.flow_from_directory(
        args.train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )

    val_data = val_datagen.flow_from_directory(
        args.train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )

    model = build_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    # Callbacks
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    cb_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2),
        callbacks.ModelCheckpoint(os.path.splitext(args.model_out)[0] + '_best.h5', save_best_only=True)
    ]

    steps_per_epoch = max(1, train_data.samples // batch_size)
    validation_steps = max(1, val_data.samples // batch_size)

    model.fit(
        train_data,
        epochs=epochs,
        validation_data=val_data,
        callbacks=cb_list,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )

    # Save final model
    model.save(args.model_out)
    print(f"✅ Model saved as {args.model_out}")


if __name__ == '__main__':
    main()
