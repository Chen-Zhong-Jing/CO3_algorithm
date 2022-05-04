import argparse
from pack.parser_utils import str2bool
from cifar10_simulations import run_experiment

# -----------------------------------------------------------------------------
"Parser Arguments"
def parse_args():
    parser = argparse.ArgumentParser(description='Variables for Cifar10 Training')

    'Model Details'
    parser.add_argument('--batch-size', type=int, default=64, help='Batch Size for Training and Testing')
    parser.add_argument('--epoch-num', type=int, default=150, help='End Iterations for Training')
    
    parser.add_argument('--optimizer', type=str, default='SGD', help='Optimizer to use. Can be either SGD or Adam')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='Learning Rate for model')
    parser.add_argument('--momentum', type=float, default=0, help='Momentum for model')
    
    parser.add_argument('--model-name', type=str, default='ResNet50V2', help='Model to Load for Training')
    
    'Print and Plot'
    parser.add_argument('--plot-state', type=str2bool, default='False', help='Whether to plot the results')
    parser.add_argument('--verbose-epoch', type=str2bool, default='True', help='Whether to print the accuracy results on a per batch basis')
    parser.add_argument('--verbose-batch', type=str2bool, default='False', help='Whether to print the accuracy results on a per batch basis')
    
    'Save Details'
    parser.add_argument('--filepath', type=str, default='D:/simulations/Cifar10/', help='Path used for saving the gradients and statistics')
    parser.add_argument('--trial-start-idx', type=int, default=0, help='Simulation start index')
    parser.add_argument('--trial-end-idx', type=int, default=2, help='Simulation end index')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    
    for i0 in range(args.trial_start_idx, args.trial_end_idx):
        args.trial_num = i0    
        run_experiment(args)