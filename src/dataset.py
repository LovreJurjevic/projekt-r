import tensorflow as tf
from sklearn.model_selection import train_test_split
from src import data
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def load_and_preprocess_image(path, label, image_height, image_width):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    
    # Resize produces floats
    img = tf.image.resize(img, (image_height, image_width))
    
    img = tf.cast(img, tf.uint8)
    
    return img, label

def create_dataset(paths, labels, image_height, image_width, batch_size, is_for_training=False):
    # Use list() to ensure the tensor slice is stable
    ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))

    ds = ds.map(
        lambda p, l: load_and_preprocess_image(p, l, image_height, image_width), 
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    ds = ds.cache() 

    if is_for_training:                         
        ds = ds.shuffle(buffer_size=len(paths))
    
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def create_datasets(image_height, image_width, batch_size):
    paths, labels = data.extract_data()

    # Stratified split 80/20
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        paths, 
        labels, 
        test_size=0.2, 
        random_state=42, 
        shuffle=True,
        stratify=labels
    )

    # Stratified split train/val
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

    # Create optimized datasets
    train_ds = create_dataset(train_paths, train_labels, image_height, image_width, batch_size, is_for_training=True)
    val_ds = create_dataset(val_paths, val_labels, image_height, image_width, batch_size)
    test_ds = create_dataset(test_paths, test_labels, image_height, image_width, batch_size)

    class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
    )

    class_weight_dict = {
        int(cls): weight
        for cls, weight in zip(np.unique(train_labels), class_weights)
    }

    return train_ds, val_ds, test_ds, class_weight_dict