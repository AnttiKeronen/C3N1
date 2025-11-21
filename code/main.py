import argparse
from utils.train_eval_helper import *

parser = argparse.ArgumentParser()

parser.add_argument('--a_note', type=str, default=None)
parser.add_argument('--dataset', type=str, default='weibo')
parser.add_argument('--seed', type=int, default=777)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=2e-6)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--conv_out', type=int, default=64)
parser.add_argument('--crop_num', type=int, default=5)   # NEW: 5 crops, not 6
parser.add_argument('--st_num', type=int, default=31)
parser.add_argument('--dropout_p', type=float, default=0)
parser.add_argument('--layer_num', type=int, default=8)
parser.add_argument('--conv_kernel', nargs='+', type=int, default=[1, 2, 3])
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--multi_gpu', type=bool, default=False)
parser.add_argument('--name', type=str, default='default')
parser.add_argument('--model', type=str, default='new')
parser.add_argument('--finetune', type=lambda x: x.lower() == 'true', default=False)
parser.add_argument('--lr_scheduler', type=lambda x: x.lower() == 'true', default=False)
parser.add_argument('--early_stopping', type=lambda x: x.lower() == 'true', default=False)
parser.add_argument('--loadpath', type=str, default=False)
parser.add_argument('--checkpoint', type=str, default=False)

args = parser.parse_args()
print(args)

set_random_seed(args.seed, deterministic=False)

# ---------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------
import torch
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from utils.data_loader_new import *
from models import *
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import os
import sys

# ---------------------------------------------------------------------
# FIXED LOCAL WINDOWS PATHS
# ---------------------------------------------------------------------
if args.dataset == "weibo":
    processed_dir = r"C:\Users\keron\OneDrive\Työpöytä\C3N\data\weibo\processed"
    save_dir_root = r"C:\Users\keron\OneDrive\Työpöytä\C3N\data\weibo\save"

    df_columns = ['original_post', 'label', 'image_id', 'post_id']

    df_train = pd.DataFrame(
        np.load(os.path.join(processed_dir, "train_EANN_frozen.npy"), allow_pickle=True),
        columns=df_columns
    )
    df_valid = pd.DataFrame(
        np.load(os.path.join(processed_dir, "valid_EANN_frozen.npy"), allow_pickle=True),
        columns=df_columns
    )
    df_test = pd.DataFrame(
        np.load(os.path.join(processed_dir, "test_EANN_frozen.npy"), allow_pickle=True),
        columns=df_columns
    )

    # NEW — only needed inputs:
    crop_input = np.load(os.path.join(processed_dir, "clip_image_preprocess.npy"), allow_pickle=True).item()
    text_input = np.load(os.path.join(processed_dir, "clean_text_inputs.npy"), allow_pickle=True).item()

    # old file removed
    n_words = None

else:
    raise NotImplementedError("Twitter pipeline not updated yet.")

# ---------------------------------------------------------------------
# DATASETS
# ---------------------------------------------------------------------
train_dataset = FakeNewsDataset(
    df_train, args.crop_num, args.st_num, args.dataset,
    n_words=n_words,
    crop_input=crop_input,
    text_input=text_input
)

valid_dataset = FakeNewsDataset(
    df_valid, args.crop_num, args.st_num, args.dataset,
    n_words=n_words,
    crop_input=crop_input,
    text_input=text_input
)

test_dataset = FakeNewsDataset(
    df_test, args.crop_num, args.st_num, args.dataset,
    n_words=n_words,
    crop_input=crop_input,
    text_input=text_input
)

# Save directory
save_dir = os.path.join(save_dir_root, args.name)
os.makedirs(save_dir, exist_ok=True)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

print("train size:", len(train_dataset))
print("valid size:", len(valid_dataset))

# ---------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------
model = C3N(args).to(args.device)

if args.checkpoint:
    valid_loss = load_checkpoint(args.checkpoint, model, args)
else:
    valid_loss = float("Inf")

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=args.lr
)

# ---------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------
@torch.no_grad()
def compute_test(model_, loader):
    model_.eval()
    loss_test = 0.0
    out_log = []

    for data in loader:
        # Move only tensor fields to device
        for k, v in data.items():
            if torch.is_tensor(v):
                data[k] = v.to(args.device)

        out = model_(data)

        out_log.append([F.softmax(out, dim=1), data['label']])
        loss_test += F.nll_loss(out, data['label']).item()

    return out_log, loss_test


def train(model, optimizer, train_loader, valid_loader, valid_loss):
    print("Start Training!")
    best_acc = 0

    for epoch in tqdm(range(args.epochs), file=sys.stdout):
        model.train()
        out_log = []
        loss_train = 0.0

        for data in train_loader:
            # Move only tensors to device (crop_input + label)
            for k, v in data.items():
                if torch.is_tensor(v):
                    data[k] = v.to(args.device)

            out = model(data)
            loss = F.nll_loss(out, data['label'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            out_log.append([F.softmax(out, dim=1), data["label"]])

        # Train metrics
        eval_train = eval_classification_report(out_log)

        # Validation
        out_log_val, loss_val = compute_test(model, valid_loader)
        eval_val = eval_classification_report(out_log_val)

        print(
            f"Epoch {epoch+1} | "
            f"Train Acc: {eval_train['accuracy']:.4f} | "
            f"Val Acc: {eval_val['accuracy']:.4f}"
        )

        # Save best model
        if eval_val["accuracy"] > best_acc:
            best_acc = eval_val["accuracy"]
            save_checkpoint(os.path.join(save_dir, "model.pt"), model, best_acc)

    print("Finished Training!")
