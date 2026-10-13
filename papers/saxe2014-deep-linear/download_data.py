"""Download the two MNIST training IDX files required for Fig. 4."""

import hashlib
import urllib.request
from pathlib import Path


FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
}
BASE = "https://ossci-datasets.s3.amazonaws.com/mnist/"


def main():
    target = Path("data/MNIST/raw")
    target.mkdir(parents=True, exist_ok=True)
    for name, expected_md5 in FILES.items():
        path = target / name
        if not path.exists():
            print(f"Downloading {name} ...", flush=True)
            urllib.request.urlretrieve(BASE + name, str(path))
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest != expected_md5:
            raise RuntimeError(f"MD5 mismatch for {path}: {digest}")
        print(f"Verified {path} (MD5 {digest})")


if __name__ == "__main__":
    main()
