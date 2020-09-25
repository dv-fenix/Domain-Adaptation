import os
import numpy as np
import torch
import torchvision as tv
import sys
import math
import torch.nn as nn
import torchvision.transforms as tvtf
import torch.optim as optim
from torch.autograd import Function
from torch.utils.tensorboard import SummaryWriter
from Data import get_dataloader
from train_utils import evaluate
from model import DANN

# Set seed
torch.manual_seed(1)

# Hyper Params
IMAGE_DIM = 28
BATCH_SIZE = 500
DIR = os.path.expanduser('~/.pytorch-datasets')
download = True
LR = 1e-3
EPOCHS = 1

# Get Data
source_dataloader, target_dataloader = get_dataloader(is_train=True, BATCH_SIZE, download, IMAGE_DIM, DIR)
source_eval_dataloader = get_dataloader(is_train=False, BATCH_SIZE, download, IMAGE_DIM, DIR)
max_batches = min(len(source_dataloader), len(target_dataloader))

# Instantiate Dicts
i = 0
labels_loss = []
source_domain_loss = []
target_domain_loss = []
batch = []

# Instantiate Tensor Board Writer
writer = SummaryWriter()

# Instantiate Model
model = DANN(image_dim=28)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load Optimizer and Loss Functions
optimizer = optim.Adam(model.parameters(), LR)

# instantiate 2 loss functions
class_loss_criterion = nn.CrossEntropyLoss()
domain_loss_criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1:04d} / {EPOCHS:04d}', end='\n=================\n')
    source_domain_iterator = iter(source_dataloader)
    target_domain_iterator = iter(target_dataloader)

    for batch__ in range(max_batches):
        optimizer.zero_grad()
        # Mark Training Progress
        progress = (batch__ + epoch * max_batches) / (EPOCHS * max_batches)
        completion_lambda = 2. / (1. + np.exp(-10 * progress)) - 1

        # Train on source domain
        X_source, y_source = next(source_domain_iterator)
        X_source = X_source.cuda()
        y_source = y_source.cuda()
        class_labels_pred, source_domain_pred = model(X_source, completion_lambda)
        loss_source_label = class_loss_criterion(class_labels_pred, y_source)

        labels_loss.append(loss_source_label)

        # Extract training data for target domain
        X_target, _ = next(target_domain_iterator)  # ignore domain labels, to be generated
        X_target = X_target.cuda()
        # Generate Domain Labels
        y_source_domain = torch.zeros(BATCH_SIZE, dtype=torch.long).cuda()
        y_target_domain = torch.ones(BATCH_SIZE, dtype=torch.long).cuda()

        # Train on target domain
        _, target_domain_pred = model(X_target, completion_lambda)

        # Calculate domain loss
        loss_source_domain = domain_loss_criterion(source_domain_pred, y_source_domain)
        loss_target_domain = domain_loss_criterion(target_domain_pred, y_target_domain)
        domain_loss__ = loss_source_domain + loss_target_domain

        source_domain_loss.append(loss_source_domain)
        target_domain_loss.append(loss_target_domain)
        batch.append(batch__)

        loss = loss_source_label + domain_loss__
        loss.backward()
        optimizer.step()

        print(f'[{batch__ + 1}/{max_batches}] '
              f'class_loss: {loss_source_label.item():.4f} ' f'source_domain_loss: {loss_source_domain.item():.4f} '
              f't_domain_loss: {loss_target_domain.item():.4f} ' f'lambda: {completion_lambda:.3f} '
              )

        writer.add_scalar('Class Loss', loss_source_label, batch__)
        writer.add_scalars(f'Domain_Loss', {
            'Source Loss': source_domain_loss[batch__],
            'Target Loss': target_domain_loss[batch__]
        }, batch__)

    acc_source = evaluate(source_eval_dataloader)
    writer.add_scalars(f'Source Domain Accuracy', {
        'Source': acc_source},
                       epoch)
    i += 1
writer.flush()