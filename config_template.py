PATH_TO_DATA = "PATH_TO_YOUR_DATA"

BATCH_SIZE = 32
LEARNING_RATE = 0.001
MOMENTUM = 0.9
NUM_EPOCHS = 10

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
SCHEDULER = CosineAnnealingLR

WEIGHT_DECAY = 1e-2

TRAIN_AUGMENTATION_TYPE = "advanced"
VALID_AUGMENTATION_TYPE = None
