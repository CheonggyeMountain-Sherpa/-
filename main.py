import os
import argparse
import numpy as np
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# from model import Model
from dataset import GetDataset
from loss import Criterion
from train import train
from transformers import GPT2LMHeadModel

## time for tensorboard log
from datetime import datetime

from early_stopping import EarlyStopping


device = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_args_parser():
    parser = argparse.ArgumentParser('NLP korean', add_help=False)

    # choose model version
    # larger the number larger the number of param choose from 0 to 7
    parser.add_argument('--model', default=0, type=int)
    
    # hyperparameters
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--sgd', default=False, type=bool)
    parser.add_argument('--num_workers', default=5, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--scheduler', default='cosine', type=str)

    # resume
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--checkpoint', default='', type=str)

    # seed
    parser.add_argument('--seed', default=42, type=int)

    # early-stopping
    parser.add_argument('--es', default=True, type=bool)
    parser.add_argument('--patience', default=4, type=int)
    
    return parser


# 하나의 seed로 고정
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if device == "cuda:0":
        torch.cuda.manual_seed(seed)  # type: ignore
        torch.backends.cudnn.deterministic = True  # type: ignore
        torch.backends.cudnn.benchmark = True      


def main(args):
    if not os.path.isdir(os.path.join(os.getcwd(), 'checkpoints')):
        os.mkdir(os.path.join(os.getcwd(), 'checkpoints'))

    # Convert time zone
    now = datetime.now()
    n_time = now.strftime("%m_%d_%H:%M")

    # Loading traindataset
    train_set = GetDataset(task='CoLA', phase='train')
    validation_set = GetDataset(task='CoLA', phase='validation')

    # make dataloader
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    validation_dataloader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Model
    model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
    model.to(device)
    model.train()

    # Optimizer
    if args.sgd:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)
    
    if args.scheduler == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.001)
    elif args.scheduler == 'multiply':
        lmbda = lambda epoch: 0.98739
        lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)

    min_val_loss = 1e9
    global_step = 0

    for epoch in range(args.epochs):
        # training
        print(f"Epoch {epoch} training")
        min_val_loss, avg_loss, val_avg_loss = train(model, train_dataloader, validation_dataloader, optimizer, epoch, device, min_val_loss, lr_scheduler, early_stopping, n_time)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Classification', parents=[get_args_parser()])
    args = parser.parse_args()
    seed_everything(args.seed)
    main(args)