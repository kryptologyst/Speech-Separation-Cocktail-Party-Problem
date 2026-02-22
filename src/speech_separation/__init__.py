"""Speech Separation Package - Modern implementation for cocktail party problem."""

__version__ = "0.1.0"
__author__ = "AI Research Team"

from .models import ConvTasNet, DPRNN
from .data import SpeechSeparationDataset, SyntheticDataGenerator
from .features import MelSpectrogramExtractor, STFTExtractor
from .losses import SISDRLoss, MultiScaleLoss
from .metrics import SISDRMetric, PESQMetric, STOIMetric
from .train import Trainer
from .eval import Evaluator

__all__ = [
    "ConvTasNet",
    "DPRNN", 
    "SpeechSeparationDataset",
    "SyntheticDataGenerator",
    "MelSpectrogramExtractor",
    "STFTExtractor",
    "SISDRLoss",
    "MultiScaleLoss",
    "SISDRMetric",
    "PESQMetric", 
    "STOIMetric",
    "Trainer",
    "Evaluator",
]
