# Main executable
# Chequear https://cs231n.github.io/neural-networks-3/#baby
import argparse
import logging
import pickle
import torch
from torch.utils.tensorboard import SummaryWriter
import time
from itertools import product
import numpy as np

import settings
import models
import dataload
import tools

parser = argparse.ArgumentParser()
parser.add_argument(
    "--network",
    type=str,
    nargs='?',
    choices=['mine', 'efficient', 'dense'],
    help='Choose the neural network: mine (very small), efficient (small) or dense (big). Default is efficient',
    default='efficient',
)
parser.add_argument(
    "--pretrained",
    type=bool,
    nargs='?',
    help='If True, use pretrained model. Default is False',
    default=False,
)
parser.add_argument(
    "--finetune",
    type=bool,
    nargs='?',
    help='To choose type of transfer learning. If True, use finetuning, else use feauture extraction. Default is False',
    default=False
)
parser.add_argument(
    "--progress",
    type=bool,
    nargs='?',
    help='If True, show the progress of training the model. Default is False',
    default=False,
)
parser.add_argument(
    "--num_workers",
    type=int,
    nargs='?',
    help='The number of workers in Dataloader. Default is 12',
    default=12,
)
parser.add_argument(
    "--batch_size",
    type=int,
    nargs='?',
    help='The batch size in Dataloader. Default is 250',
    default=250,
)
parser.add_argument(
    "--max_epochs",
    type=int,
    nargs='?',
    help='Number of epochs to train the model. Default is 200',
    default=200,
)
parser.add_argument(
    "--lr",
    type=float,
    nargs='?',
    help='Learning rate. Default is 0.001',
    default=1e-3
)
task_subparser = parser.add_subparsers(help='Task to perform', dest='task')
train_parser = task_subparser.add_parser("train")

if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("WARNING: Running on CPU")
    class_names, train_ds, test_ds = dataload.load_images()
    model = models.MyNeuralNetwork()
    if args.pretrained:
        if args.finetune:
            suffix = "pretrained_finetune"
        else:
            suffix = "pretrained_featextract"
    else:
        suffix = ''
    model_name = f"nn_model_{args.network}_{suffix}_lr{args.lr}dataaug.pth"

    if args.task == "train":
        logging.info(f"Training the neural network model")
        if args.num_workers == 0:
            print('Single-process loading')
        elif args.num_workers > 0:
            print('Multi-process loading')
        writer = SummaryWriter('runs/train/' + args.network + suffix + f'lr{args.lr}dataaug')
        writer.add_text('count', f"Total image count: {len(train_ds)+len(test_ds)}")
        writer.flush()
        since = time.time()
        num_each = model.train(train_ds, test_ds, device, class_names, args.num_workers, args.batch_size, args.max_epochs, writer, args.progress, args.pretrained, args.finetune, args.network, model_name, args.lr)
        time_elapsed = time.time() - since
        writer.add_figure('Label counts', tools.plot_each(num_each))
        writer.add_text('train', f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        writer.close()
        model.save(settings.MODELS_PATH / model_name)
        print(f"Model succesfully saved in {settings.MODELS_PATH / model_name}.")
        print("See the analysis in Tensorboard.")