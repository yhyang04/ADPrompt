import numpy as np
import random
import argparse
import torch

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser(description='Downstream task: node classification')
    parser.add_argument('--dataset_name', type=str, default='credit', help='Dataset name (default: credit)')
    parser.add_argument('--pretrain_task', type=str, default='DGI', help='Pre-training task (default: DGI)')
    parser.add_argument('--shots', type=int, default=50, help='Number of shots (default: 50)')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension (default: 128)')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for training (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--lambda', type=float, default=5.0, help='balance between Supervised Loss and Adversarial Loss')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs (default: 300)')
    parser.add_argument('--adv_alpha', type=float, default=0.5, help='Alpha for Gradient Reversal Layer (default: 0.5)')
    args = parser.parse_args()
    return args
