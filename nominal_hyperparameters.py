from dataclasses import dataclass

@dataclass
class Hyperparameters:
    # define training hyperparameters
    INIT_LR: float = 1e-3
    BATCH_SIZE: int = 64
    EPOCHS: int = 10
    # define the train and val splits
    TRAIN_SPLIT: float = 0.75
    VAL_SPLIT: float = 1 - TRAIN_SPLIT
