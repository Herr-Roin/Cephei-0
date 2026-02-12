import os
import gzip
import urllib.request
import urllib.error
import numpy as np

BASE_URLS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://yann.lecun.com/exdb/mnist/",
]

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}

DATA_DIR = "data"
OUT_TRAIN = os.path.join(DATA_DIR, "mnist_train.npz")
OUT_TEST  = os.path.join(DATA_DIR, "mnist_test.npz")

def download_if_needed(filename: str) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        return filepath

    last_err = None
    for base in BASE_URLS:
        url = base + filename
        try:
            print(f"Downloading {filename} from {base} ...")
            urllib.request.urlretrieve(url, filepath)
            return filepath
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            last_err = e
            # If a partial file was created, remove it
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except OSError:
                    pass
            print(f"  failed: {e}")

    raise RuntimeError(f"Could not download {filename} from any mirror. Last error: {last_err}")

def load_images(filepath: str) -> np.ndarray:
    with gzip.open(filepath, "rb") as f:
        _magic = int.from_bytes(f.read(4), "big")
        num = int.from_bytes(f.read(4), "big")
        rows = int.from_bytes(f.read(4), "big")
        cols = int.from_bytes(f.read(4), "big")
        data = f.read()
    images = np.frombuffer(data, dtype=np.uint8).reshape(num, rows * cols)
    return images.astype(np.float32) / 255.0  # (N, 784) in [0,1]

def load_labels(filepath: str) -> np.ndarray:
    with gzip.open(filepath, "rb") as f:
        _magic = int.from_bytes(f.read(4), "big")
        num = int.from_bytes(f.read(4), "big")
        data = f.read()
    labels = np.frombuffer(data, dtype=np.uint8)
    if labels.shape[0] != num:
        raise ValueError("Label file length mismatch")
    return labels.astype(np.int64)  # (N,)

def main():
    train_images_path = download_if_needed(FILES["train_images"])
    train_labels_path = download_if_needed(FILES["train_labels"])
    test_images_path  = download_if_needed(FILES["test_images"])
    test_labels_path  = download_if_needed(FILES["test_labels"])

    print("Loading training set...")
    X_train = load_images(train_images_path)
    Y_train = load_labels(train_labels_path)

    print("Loading test set...")
    X_test = load_images(test_images_path)
    Y_test = load_labels(test_labels_path)

    np.savez(OUT_TRAIN, X=X_train, Y=Y_train)
    np.savez(OUT_TEST,  X=X_test,  Y=Y_test)

    print("Saved:")
    print(" ", OUT_TRAIN, "X:", X_train.shape, "Y:", Y_train.shape)
    print(" ", OUT_TEST,  OUT_TEST,  "X:", X_test.shape,  "Y:", Y_test.shape)

if __name__ == "__main__":
    main()