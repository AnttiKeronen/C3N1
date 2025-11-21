import torch
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.metrics import classification_report
import csv
import torch.nn as nn


# ============================================================
# SAVE / LOAD CHECKPOINTS
# ============================================================

def save_checkpoint(save_path, model, valid_loss):
    """Save model checkpoint."""
    if save_path is None:
        return

    state_dict = {
        'model_state_dict': model.state_dict(),
        'valid_loss': valid_loss
    }
    torch.save(state_dict, save_path)
    print(f"Model saved to ==> {save_path}")


def load_checkpoint(load_path, model, args):
    """Load model checkpoint."""
    if load_path is None:
        return None

    state_dict = torch.load(load_path, map_location=args.device)
    print(f"Model loaded from <== {load_path}")

    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict.get('valid_loss', None)


def load_partial_dict(load_path, model, args):
    """Load model weights (non-strict) for transfer learning."""
    state_dict = torch.load(load_path, map_location=args.device)
    model.load_state_dict(state_dict['model_state_dict'], strict=False)
    print(f"Model partial parameters loaded from <== {load_path}")


# ============================================================
# TRAINING PLOTS
# ============================================================

def draw_fig_loss(train_loss, val_loss, epochs, save_dir):
    plt.cla()
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Valid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, "train_valid_loss.jpg"), dpi=300)


def draw_fig_acc(train_acc, val_acc, epochs, save_dir):
    plt.cla()
    plt.plot(epochs, train_acc, label="Train Acc.")
    plt.plot(epochs, val_acc, label="Valid Acc.")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, "train_valid_acc.jpg"), dpi=300)


# ============================================================
# RANDOM SEED SETUP
# ============================================================

def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# EVALUATION HELPERS
# ============================================================

def eval_classification_report(out_log, do_print=False):
    """
    out_log contains batches: [(logits, labels), ...]
    """
    pred_y_list = []
    y_list = []

    for logits, labels in out_log:
        preds = logits.detach().cpu().numpy().argmax(axis=1).tolist()
        true = labels.detach().cpu().numpy().tolist()
        pred_y_list.extend(preds)
        y_list.extend(true)

    out = classification_report(
        y_list,
        pred_y_list,
        labels=[1, 0],
        target_names=['Fake', 'True'],
        digits=4,
        output_dict=(not do_print)
    )

    if do_print:
        print(out)

    return out


def create_out_csv(root, name):
    path = os.path.join(root, name + "_out.csv")
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Args', 'Accuracy', 'Mac.F1',
                         'F-P', 'F-R', 'F-F1',
                         'T-P', 'T-R', 'T-F1'])


def append_out_csv(root, name, out, args):
    path = os.path.join(root, name + "_out.csv")

    with open(path, 'a+', newline='') as f:
        writer = csv.writer(f)
        row = [
            args,
            f"{out['accuracy']:.4f}",
            f"{out['macro avg']['f1-score']:.4f}",
            f"{out['Fake']['precision']:.4f}",
            f"{out['Fake']['recall']:.4f}",
            f"{out['Fake']['f1-score']:.4f}",
            f"{out['True']['precision']:.4f}",
            f"{out['True']['recall']:.4f}",
            f"{out['True']['f1-score']:.4f}"
        ]
        writer.writerow(row)

    print(f"Out saved to ==> {path}")


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

class LRScheduler:
    """Reduces LR when validation accuracy plateaus."""

    def __init__(self, optimizer, patience=4, min_lr=1e-6, factor=0.5):
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',      # maximize accuracy
            patience=patience,
            factor=factor,
            min_lr=min_lr,
            verbose=True
        )

    def __call__(self, val_acc):
        self.lr_scheduler.step(val_acc)


# ============================================================
# EARLY STOPPING (FIXED)
# ============================================================

class EarlyStoppingAcc:
    """Stop training when validation accuracy stops improving."""

    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_acc = -float("inf")   # FIXED (was +inf before!)
        self.early_stop = False

    def __call__(self, val_acc):
        if val_acc > self.best_acc + self.min_delta:
            self.best_acc = val_acc
            self.counter = 0
        else:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                print("INFO: Early stopping triggered")
                self.early_stop = True
