import os
import collections
import numpy as np
import pandas as pd
from trainers.abstract_trainer import AbstractTrainer
from utils import AverageMeter,setSeed, starting_logs


class Trainer(AbstractTrainer):
    def __init__(self, args):
        super().__init__(args)

        self.results_columns = []
        
    def train(self):
        
        
        setSeed(1234) # set seed for reproducibility
        
        # table with metrics
        self.results_columns = ["scenario", 'src_rmse',' src_r2', 'tgt_rmse', 'tgt_r2']
        table_results = pd.DataFrame(columns=self.results_columns)
        
        for src_id, tgt_id, tst_id in self.dataset_configs.scenarios:
            
            # Logging
            self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.da_method, self.exp_log_dir, src_id, tgt_id)
            
            # Average meters
            self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())
            
            # Loading data
            self.load_data(src_id, tgt_id)
            
            # Initialize model
            self.initialize_model()
            
            # Train the domain adaptation algorithm
            self.last_model = self.model.update(self.src_dl, self.tgt_dl, self.loss_avg_meters, self.logger)
            
            # Calculate risks and metrics on both source and target data
            metrics = self.calculate_metrics()
            
            # Append results to tables
            scenario = f"{src_id}_to_{tgt_id}"
            table_results = self.append_results_to_tables(table_results, scenario, metrics)
        
        # Calculate and append mean and std to tables
        table_results = self.add_mean_std_table(table_results, self.results_columns)
        
        # Save tables to file
        self.save_tables_to_file(table_results, 'results')
            
        
        
    def test(self):
        
        # Results dataframes
        self.results_columns = ["scenario", 'tgt_rmse',' tgt_r2']
        last_results = pd.DataFrame(columns=self.results_columns)
        
        # Cross-domain scenarios
        for src_id, tgt_id, tst_id in self.dataset_configs.scenarios:
            # fixing random seed
            setSeed(1234)

            # Logging
            self.scenario_log_dir = os.path.join(self.exp_log_dir, src_id + "_to_" + tgt_id + "_on_" + tst_id)

            self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

            # Load data
            self.load_data(src_id, tgt_id, tst_id)

            # Build model
            self.initialize_model()

            # Load chechpoint 
            last_chk = self.load_checkpoint(self.scenario_log_dir)

            # Testing the model on test data
            self.model.network.load_state_dict(last_chk)
            self.evaluate(self.tst_dl)
            last_metrics = self.calculate_metrics()
            last_results = self.append_results_to_tables(last_results, f"{src_id}_to_{tgt_id}_on_{tst_id}", last_metrics)
            

        last_scenario_mean_std = last_results.groupby('scenario')[['RMSE', 'R2Score']].agg(['mean', 'std'])


        # Save tables to file if needed
        self.save_tables_to_file(last_scenario_mean_std, 'last_results')

        # printing summary 
        summary_last = {metric: np.mean(last_results[metric]) for metric in self.results_columns[2:]}
        for summary_name, summary in [('Last', summary_last)]:
            for key, val in summary.items():
                print(f'{summary_name}: {key}\t: {val:2.4f}')