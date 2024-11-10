import argparse
import torch
from pathlib import Path
import logging
from adnn.methods.adnn_linear import AdaptiveDynamicsNeuralNetwork
from adnn.utils.data import get_dataset
from adnn.utils.train import train
from adnn.utils.eval import evaluate

def create_parser():
    parser = argparse.ArgumentParser(description='ADNN CLI tool')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Debug command
    debug_parser = subparsers.add_parser('debug', help='Run debug utilities')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--dataset', type=str, default='synthetic', choices=['synthetic', 'wave'],
                            help='Dataset to use for training')
    train_parser.add_argument('--N', type=int, default=1024, help='Dimension of the system')
    train_parser.add_argument('--num-samples', type=int, default=1000, help='Number of samples')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    train_parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    train_parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    train_parser.add_argument('--log-dir', type=str, default='runs', help='Tensorboard log directory')
    train_parser.add_argument('--model-dir', type=str, default='models', help='Model save directory')
    
    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Evaluate the model')
    eval_parser.add_argument('--model-path', type=str, required=True, help='Path to saved model')
    eval_parser.add_argument('--dataset', type=str, default='synthetic', choices=['synthetic', 'wave'],
                            help='Dataset to use for evaluation')
    eval_parser.add_argument('--N', type=int, default=1024, help='Dimension of the system')
    eval_parser.add_argument('--num-samples', type=int, default=200, help='Number of samples')
    eval_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    
    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if args.command == 'debug':
        from adnn.utils.debug import main
        main()
    
    elif args.command == 'train':
        # Create model
        model = AdaptiveDynamicsNeuralNetwork(args.N, device=device).to(device)
        
        # Get data
        train_loader, val_loader = get_dataset(args.dataset, args.num_samples, args.N, args.batch_size)
        
        # Training configuration
        config = {
            'device': device,
            'num_epochs': args.epochs,
            'learning_rate': args.lr,
            'log_dir': args.log_dir,
            'model_dir': args.model_dir,
            't_span': [0.0, 1.0],
            'dt': 0.01,
            'clip_grad': True,
            'max_grad_norm': 1.0,
            'log_interval': 10,
            'eval_interval': 2
        }
        
        # Train model
        train(model, train_loader, val_loader, config)
    
    elif args.command == 'eval':
        # Create model and load weights
        model = AdaptiveDynamicsNeuralNetwork(args.N, device=device).to(device)
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Get data
        _, val_loader = get_dataset(args.dataset, args.num_samples, args.N, args.batch_size)
        
        # Evaluate
        criterion = torch.nn.MSELoss()
        val_loss = evaluate(model, val_loader, [0.0, 1.0], 0.01, criterion, device)
        print(f'Evaluation Loss: {val_loss:.4f}')
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main()