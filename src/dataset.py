import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from src import data

data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomFlip("vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
    ])

def load_and_preprocess_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    #img = tf.image.rgb_to_grayscale(img)
    img = tf.image.resize(img, (64,64))
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

def create_dataset(paths, labels, is_for_training = False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    if is_for_training:
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    ds = ds.batch(16).prefetch(tf.data.AUTOTUNE)
    return ds


def create_datasets():
    paths, labels = data.extract_data()

    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        paths, 
        labels, 
        test_size=0.2, 
        random_state=42, 
        shuffle=True,
        stratify=labels
    )

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, 
        train_val_labels, 
        test_size=0.25, 
        random_state=42, 
        shuffle=True,
        stratify=train_val_labels
    )

    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    print(f"Testing samples: {len(test_paths)}")

    
    train_ds = create_dataset(train_paths, train_labels, is_for_training=True)
    val_ds = create_dataset(val_paths, val_labels)
    test_ds = create_dataset(test_paths, test_labels)

    return train_ds, val_ds, test_ds