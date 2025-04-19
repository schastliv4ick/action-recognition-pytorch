PATH_TO_DATA = "D:\\yandex-ml-2025\\data\\human_poses_data_reduced"

BATCH_SIZE = 32
LEARNING_RATE = 0.001
MOMENTUM = 0.9
NUM_EPOCHS = 3

WEIGHT_DECAY = 1e-2

TRAIN_AUGMENTATION_TYPE = "basic"
VALID_AUGMENTATION_TYPE = None

MODEL = "PoseCNNsc_13_24_35" # from models/stage1.py import PoseCNNsc_13_24_35