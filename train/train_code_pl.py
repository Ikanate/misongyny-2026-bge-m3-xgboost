import os

import pickle
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModel,
                          get_linear_schedule_with_warmup)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

#Actual libraries
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

## (PENDING) Libreries with the model
### Example -> from models.baseModel import BGETaggingModel
from models.bge_m3_base import MultiTaskBGE 

class BGE_LightningModule(pl.LightningModule):
    def __init__(self.**model_kwargs):
        super().__init__()

        self.model = BGETaggingModel()

    def forward(self, input_data):
        return self.model(input_data)
