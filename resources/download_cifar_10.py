# %%
# download the CIPHER dataset from the UCI repository and preprocess it for ImageFolder format

import tensorflow as tf
import os

# %%
cifar_10 = tf.keras.datasets.cifar10.load_data()

# %%

path_to_cifar_10 = os.path.join(os.getcwd(), "downloaded", "cifar_10")
os.makedirs(path_to_cifar_10, exist_ok=True)

# %%

# store the training data in a folders where each folder is a class
classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
for i, class_name in enumerate(classes):
    os.makedirs(
        os.path.join(path_to_cifar_10, "all_png", class_name), exist_ok=True
    )
# %%
# store the training data in a folders where each folder is a class
for i, class_name in enumerate(classes):
    for j, image in enumerate(cifar_10[0][0][cifar_10[0][1].flatten() == i]):
        tf.io.write_file(
            os.path.join(path_to_cifar_10, "all_png", class_name, f"{j}.png"),
            tf.image.encode_png(image),
        )


# %%
