import argparse
from trainers.train import Trainer

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    
    # ========== Phase ==========
    parser.add_argument('--phase', type=str, default='train', help='train or test')
    
    # ==========  Experiments Name ==========
    parser.add_argument('--save_dir', default='experiments', type=str, help='Directory containing all experiments')
    parser.add_argument('--exp_name', default='exp1', type=str, help='experiment name')
    
    # ========== Data ==========
    parser.add_argument('--dataset', type=str, default='Airbox', help='dataset name')
    
    # ========== DA method ==========
    parser.add_argument('--da_method', type=str, default='AirDA', help='Algorithm : (NO_ADAPT, DANN, DIRT, MMDA)')
    
    # ========== Backbone ==========
    parser.add_argument('--backbone', type=str, default='GRU', help='Backbone : (GRU)')
    
    args = parser.parse_args()
    
    trainer = Trainer(args)
    
    if args.phase == 'train':
        trainer.train()
    elif args.phase == 'test':
        trainer.test()
    
    