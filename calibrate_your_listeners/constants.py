
from calibrate_your_listeners.src.datasets import (
    shapeworld
)
import os

ROOT_DIR=os.getcwd()
MAIN_REPO_DIR=os.getcwd()
DROPOUT_LISTENER_MODEL_DIR=os.path.join(MAIN_REPO_DIR, "src/models/checkpoints")
NORMAL_LISTENER_MODEL_DIR=os.path.join(MAIN_REPO_DIR, "src/models/checkpoints")

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2

MAX_SEQ_LEN=10
EPS=1e-5

NAME2DATASETS = {
    'shapeworld': shapeworld.Shapeworld
}
