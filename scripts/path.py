"""
Paths and procedures to manage archives and directories.
"""
import os
import sys
import zipfile

# Directories paths
DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(DIRNAME)
DATA_DIR = os.path.join(ROOT_DIR, 'data')
OUT_DIR = os.path.join(ROOT_DIR, 'out')
SRC_DIR = os.path.join(ROOT_DIR, 'src')

# Files paths
DATA_TEST_PATH = os.path.join(DATA_DIR, 'test.csv')
DATA_TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')


def add_src_to_path() -> None:
    """Adds the "src" directory to the path."""
    sys.path.append(SRC_DIR)


def create_out_dir() -> None:
    """Creates the "out" directory if needed."""
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)


def extract_archives() -> None:
    """Extracts the archives in the data directory if needed."""
    for csv_filename in (DATA_TEST_PATH, DATA_TRAIN_PATH):
        if not os.path.exists(csv_filename):
            zip_filename = csv_filename + '.zip'
            with zipfile.ZipFile(zip_filename, 'r') as zf:
                zf.extractall(DATA_DIR)
