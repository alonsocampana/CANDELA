#!/usr/bin/env python3.9

import pandas as pd
import numpy as np
import pickle
import torch
import json
from torch import nn
from torch.nn import functional as F
import os
import gc
import base64
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
import argparse
from filelock import FileLock
import sys
from utils import JSONLogger
from GATF import prepare_dataloaders, get_model, train_epoch, test_epoch

def train_model(config, checkpoint_dir=None):
    logger = JSONLogger()
    loss_fn = nn.MSELoss
    train_dataloader, test_dataloader = prepare_dataloaders(**config)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = get_model(**config)
    lr=config["learning_rate"]
    betas=(config["beta_1"], config["beta_2"])
    weight_decay=config["weight_decay"]
    optim = torch.optim.Adam(model.parameters(), lr, betas=betas, weight_decay=weight_decay)
    model.to(device)
    best_loss = 1
    trainin_report = {}
    for epoch in range(500):
        train_loss = train_epoch(device, loss_fn, train_dataloader, model, optim)
        test_loss = test_epoch(device, loss_fn, test_dataloader, model)
        logger(epoch, train_loss = train_loss, test_loss=test_loss)
        gc.collect()
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), "trained_models/best_GATF.pth")
    torch.save(model.state_dict(), "trained_models/last_GATF.pth")


if __name__ == "__main__":
    with open("params/optimal_config_GATF.json", "r") as f:
        config = json.load(f)
    train_model(config)
    gc.collect()
    
