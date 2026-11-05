"""Generate CIFAR-10 images of a chosen class with the trained conditional GAN.

Examples:
    python generate.py                      # 10 images of every class
    python generate.py --cls dog --n 16     # 16 dogs
    python generate.py --cls ship --seed 7 --out ships.png
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
LATENT_DIM = 128


def build_generator(latent_dim=LATENT_DIM, num_classes=10):
    """Wider/deeper conditional generator (same architecture as the notebook)."""
    noise_input = layers.Input(shape=(latent_dim,))
    label_input = layers.Input(shape=(1,), dtype="int32")

    label_embedding = layers.Embedding(num_classes, 50)(label_input)
    label_embedding = layers.Flatten()(label_embedding)
    label_map = layers.Dense(4 * 4)(label_embedding)
    label_map = layers.Reshape((4, 4, 1))(label_map)

    x = layers.Dense(4 * 4 * 511, use_bias=False)(noise_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Reshape((4, 4, 511))(x)
    x = layers.Concatenate()([x, label_map])

    def block(x, filters, name):
        x = layers.Conv2DTranspose(filters, 4, strides=2, padding="same", use_bias=False, name=f"{name}_deconv")(x)
        x = layers.BatchNormalization(name=f"{name}_bn1")(x)
        x = layers.LeakyReLU(alpha=0.2, name=f"{name}_act1")(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_refine1")(x)
        x = layers.BatchNormalization(name=f"{name}_bn2")(x)
        x = layers.LeakyReLU(alpha=0.2, name=f"{name}_act2")(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_refine2")(x)
        x = layers.BatchNormalization(name=f"{name}_bn3")(x)
        x = layers.LeakyReLU(alpha=0.2, name=f"{name}_act3")(x)
        return x

    x = block(x, 256, "block_8x8")
    x = block(x, 128, "block_16x16")
    x = block(x, 64, "block_32x32")
    out = layers.Conv2D(3, 5, padding="same", activation="tanh")(x)
    return tf.keras.Model([noise_input, label_input], out, name="cGAN_Generator_WideDeep")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default="ema_generator.weights.h5", help="path to the EMA generator weights")
    p.add_argument("--cls", default="all", help="class name (e.g. dog) or 'all'")
    p.add_argument("--n", type=int, default=10, help="images per class")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="generated.png")
    a = p.parse_args()

    names = CLASSES if a.cls == "all" else [a.cls]
    if any(n not in CLASSES for n in names):
        raise SystemExit(f"Unknown class '{a.cls}'. Choose from: {', '.join(CLASSES)}")

    gen = build_generator()
    gen.load_weights(a.weights)

    rng = np.random.default_rng(a.seed)
    fig, axes = plt.subplots(len(names), a.n, figsize=(a.n * 1.2, len(names) * 1.2), squeeze=False)
    for r, name in enumerate(names):
        noise = rng.standard_normal((a.n, LATENT_DIM)).astype("float32")
        labels = np.full((a.n, 1), CLASSES.index(name), dtype="int32")
        imgs = (gen([noise, labels], training=False).numpy() + 1) / 2  # [-1, 1] -> [0, 1]
        for c in range(a.n):
            axes[r, c].imshow(np.clip(imgs[c], 0, 1))
            axes[r, c].axis("off")
        axes[r, 0].set_title(name, fontsize=8, loc="left")
    plt.tight_layout()
    plt.savefig(a.out, dpi=150)
    print(f"Saved {a.out}")


if __name__ == "__main__":
    main()
