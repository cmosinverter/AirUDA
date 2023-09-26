import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from configs.configs import get_dataset_class
from configs.hparams import get_hparams_class
from algorithms.algorithms import get_algorithm_class
from blocks.blocks import get_backbone_class
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from dataloader.dataloader import data_generator

class AbstractTrainer(object):
    
    def __init__(self, args):
        
        # get phase
        self.phase = args.phase
        
        # get dataset path
        self.home_path =  os.getcwd()
        self.dataset = args.dataset
        self.datapath = os.path.join('datasets', self.dataset)
        
        # da method and backbone
        self.da_method = args.da_method
        self.backbone = args.backbone
        
        # get save dir and experiment name
        self.save_dir = args.save_dir
        self.exp_name = args.exp_name
        self.exp_log_dir = os.path.join(self.save_dir, self.dataset, f"{self.da_method}_{self.exp_name}")
        
        # get dataset and base model configs
        self.dataset_configs, self.hparams_class = self.get_configs()
    
        # Specify number of hparams
        self.hparams = {**self.hparams_class.alg_hparams[self.da_method],  **self.hparams_class.train_params}
        
        # training device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    
    def get_configs(self):
        dataset_class = get_dataset_class(self.dataset)
        hparams_class = get_hparams_class(self.dataset)
        return dataset_class(), hparams_class()
    
    def initialize_model(self):
        algorithm_class = get_algorithm_class(self.da_method)
        backbone_fe = get_backbone_class(self.backbone)
        
        # initialize model
        self.model = algorithm_class(backbone_fe, self.dataset_configs, self.hparams, self.device)
        self.model.to(self.device)
        
    def evaluate(self, test_loader):
        feature_extractor = self.model.feature_extractor.to(self.device)
        regressor = self.model.regressor.to(self.device)

        feature_extractor.eval()
        regressor.eval()

        total_loss, preds_list, labels_list = [], [], []
        feature_list = []

        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)

                # forward pass
                features = feature_extractor(data)
                predictions = regressor(features)
                # compute loss
                loss = mean_squared_error(predictions.cpu(), labels.cpu(), squared=False)
                total_loss.append(loss.item())
                pred = predictions.detach()

                # append predictions and labels
                preds_list.append(pred)
                labels_list.append(labels)
                feature_list.append(features)

        self.loss = torch.tensor(total_loss).mean()  # average loss
        
        
        self.full_preds = torch.cat((preds_list))
        self.full_labels = torch.cat((labels_list))
        self.full_features = torch.cat((feature_list))
        
    def visualize_feature_space(self, src_id, tgt_id):
        self.evaluate(self.src_dl)
        src_features = self.full_features.cpu()
        src_labels = self.full_labels.cpu()
        self.evaluate(self.tgt_dl)
        tgt_features = self.full_features.cpu()
        tgt_labels = self.full_labels.cpu()
        
        # pca dimensionality reduction to 2D
        pca = PCA(n_components=2)
        src_features = pca.fit_transform(src_features)
        tgt_features = pca.fit_transform(tgt_features)
        
        # plot feature space in one figure, source and target have different colors
        plt.figure(figsize=(10, 10))
        plt.scatter(src_features[:, 0], src_features[:, 1], color = 'blue', label='Source')
        plt.scatter(tgt_features[:, 0], tgt_features[:, 1], color = 'red', label='Target')
        plt.title(f"Source: {src_id} and Target: {tgt_id} feature space")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.exp_log_dir, f"{src_id}_to_{tgt_id}_features.png"))
        
    def load_data(self, src_id, tgt_id, tst_id = None):
        self.src_dl = data_generator(self.datapath, src_id, self.dataset_configs, self.hparams)
        self.tgt_dl = data_generator(self.datapath, tgt_id, self.dataset_configs, self.hparams)
        
        if tst_id:
            self.tst_dl = data_generator(self.datapath, tst_id, self.dataset_configs, self.hparams)
        
    def create_save_dir(self, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            
    def calculate_metrics(self, phase):
        
        # Calculate metrics on source and target data
        if phase == 'train':
            self.evaluate(self.src_dl)
            src_rmse = mean_squared_error(self.full_preds.cpu(), self.full_labels.cpu(), squared=False).item()
            src_r2score = r2_score(self.full_labels.cpu(), self.full_preds.cpu()).item()
            self.evaluate(self.tgt_dl)
            tgt_rmse = mean_squared_error(self.full_preds.cpu(), self.full_labels.cpu(), squared=False).item()
            tgt_r2score = r2_score(self.full_labels.cpu(), self.full_preds.cpu()).item()
            return src_rmse, src_r2score, tgt_rmse, tgt_r2score
         
        # Calculate metrics on test data
        elif phase == 'test':
            self.evaluate(self.tst_dl)
            tst_rmse = mean_squared_error(self.full_preds.cpu(), self.full_labels.cpu(), squared=False).item()
            tst_r2score = r2_score(self.full_labels.cpu(), self.full_preds.cpu()).item()
            return tst_rmse, tst_r2score
    
    def append_results_to_tables(self, table, scenario, metrics):
        # Create metrics rows
        results_row = [scenario, *metrics]

        # Concatenate new dataframes with original dataframes
        table.loc[len(table)] = results_row
        
        return table
        
    def save_tables_to_file(self, table_results, name):
        
        # Create a formatting function to format each element in the tables
        format_func = lambda x: f"{x:.4f}" if isinstance(x, float) else x
        
        # Apply the formatting function to each element in the tables
        table_results = table_results.map(format_func)
        
        table_results.to_csv(os.path.join(self.exp_log_dir,f"{name}.csv"))
        
    def add_mean_std_table(self, table, columns):
        # Calculate average and standard deviation for metrics
        avg_metrics = [table[metric].mean() for metric in columns[1:]]
        std_metrics = [table[metric].std() for metric in columns[1:]]

        # Create dataframes for mean and std values
        mean_metrics_df = pd.DataFrame([['mean',  *avg_metrics]], columns=columns)
        std_metrics_df = pd.DataFrame([['std',  *std_metrics]], columns=columns)

        # Concatenate original dataframes with mean and std dataframes
        table = pd.concat([table, mean_metrics_df, std_metrics_df], ignore_index=True)

        return table
    
    def visualize_results(self, src_id, tgt_id, tst_id):
        
        
        # plot results, source on the top and target on the bottom
        fig, axs = plt.subplots(2, 1, figsize=(15, 10))
        self.evaluate(self.src_dl)
        axs[0].plot(self.full_labels.cpu(), 'b', label='True')
        axs[0].plot(self.full_preds.cpu(), 'r', label='Predicted')
        axs[0].set_title(f"Source: {src_id}")
        axs[0].legend()
        self.evaluate(self.tgt_dl)
        axs[1].plot(self.full_labels.cpu(), 'b', label='True')
        axs[1].plot(self.full_preds.cpu(), 'r', label='Predicted')
        axs[1].set_title(f"Target: {tgt_id}")
        axs[1].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.exp_log_dir, f"{src_id}_to_{tgt_id}.png"))

        # Save the numerical results to file
        # torch.save(self.full_preds.cpu(), os.path.join(self.exp_log_dir, f"{src_id}_to_{tgt_id}_pred.pt"))
        
    def save_checkpoint(self, home_path, log_dir, last_model):
        save_dict = {
            "last": last_model
        }
        # save classification report
        save_path = os.path.join(home_path, log_dir, f"checkpoint.pt")
        torch.save(save_dict, save_path)
        
        
    
    def load_checkpoint(self, model_dir):
        checkpoint = torch.load(os.path.join(self.home_path, model_dir, 'checkpoint.pt'))
        last_model = checkpoint['last']
        return last_model