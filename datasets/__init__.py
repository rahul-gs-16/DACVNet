from .c3vd_dataset import C3VDDataset
from .own_ss_dataset import OwnSSDataset


__datasets__ = {
    "c3vd":C3VDDataset,
    "own_ss":OwnSSDataset
}
