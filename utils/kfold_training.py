import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import warnings
warnings.filterwarnings('ignore')


class KFoldTrainer:
    """
    K-Fold Cross Validation Trainer for Document Classification
    """
    
    def __init__(self, n_splits=5, shuffle=True, random_state=42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.fold_results = []
        
    def create_folds(self, dataset, labels=None):
        """
        Create K-fold splits for training
        
        Args:
            dataset: Full dataset
            labels: Labels for stratified split (if None, extract from dataset)
        """
        if labels is None:
            # Extract labels from dataset
            labels = []
            for i in range(len(dataset)):
                _, label, _ = dataset[i]
                labels.append(label)
            labels = np.array(labels)
        
        # Create stratified k-fold
        skf = StratifiedKFold(
            n_splits=self.n_splits, 
            shuffle=self.shuffle, 
            random_state=self.random_state
        )
        
        folds = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(dataset)), labels)):
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            folds.append((train_subset, val_subset, fold_idx))
        
        return folds
    
    def train_fold(self, fold_data, model_class, config, training_fn, device):
        """
        Train a single fold
        
        Args:
            fold_data: (train_subset, val_subset, fold_idx)
            model_class: Model class to instantiate
            config: Training configuration
            training_fn: Training function
            device: Training device
        """
        train_subset, val_subset, fold_idx = fold_data
        
        print(f"\n=== Training Fold {fold_idx + 1}/{self.n_splits} ===")
        print(f"Train samples: {len(train_subset)}, Val samples: {len(val_subset)}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_subset,
            batch_size=config['BATCH_SIZE'],
            shuffle=True,
            num_workers=config.get('num_workers', 4),
            pin_memory=True,
            drop_last=False
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=config['BATCH_SIZE'],
            shuffle=False,
            num_workers=config.get('num_workers', 4),
            pin_memory=True,
            drop_last=False
        )
        
        # Initialize model
        model = model_class(num_classes=config['num_classes']).to(device)
        
        # Setup optimizer, scheduler, loss function
        from utils.optimizer_factory import get_optimizer
        from utils.scheduler_factory import get_scheduler
        from utils.loss_factory import get_loss
        
        optimizer = get_optimizer(
            config['optimizer']['name'], 
            model.parameters(), 
            config['optimizer']['params']
        )
        
        scheduler = get_scheduler(
            config['scheduler']['name'], 
            optimizer, 
            config['scheduler']['params']
        )
        
        criterion = get_loss(
            config['loss']['name'], 
            config['loss']['params']
        )
        
        # Train the model
        from trainer.train_loop import training_loop
        from utils.EarlyStopping import EarlyStopping
        
        early_stopping = EarlyStopping(
            patience=config['patience'], 
            delta=config['delta'], 
            verbose=True, 
            save_path=f"fold_{fold_idx}_best.pth",
            mode='max'
        )
        
        # Create dummy logger for fold training
        class DummyLogger:
            def log_metrics(self, metrics, step=None):
                pass
            def log_confusion_matrix(self, y_true, y_pred, class_names, step=None):
                pass
            def log_failed_predictions(self, *args, **kwargs):
                pass
            def log_predictions(self, *args, **kwargs):
                pass
            def save_model(self, *args, **kwargs):
                pass
            def finish(self):
                pass
        
        logger = DummyLogger()
        
        training_args = {}
        if config.get('training_mode') == 'on_amp':
            from torch.cuda.amp import GradScaler
            training_args['scaler'] = GradScaler()
        
        # Train the model
        trained_model, best_val_accuracy = training_loop(
            training_fn,
            model, train_loader, val_loader, train_subset, val_subset,
            criterion, optimizer, device, config['EPOCHS'],
            early_stopping, logger, config['class_names'], scheduler,
            training_args
        )
        
        # Evaluate the fold
        fold_result = self.evaluate_fold(trained_model, val_loader, device, fold_idx)
        
        # Clean up
        if os.path.exists(f"fold_{fold_idx}_best.pth"):
            os.remove(f"fold_{fold_idx}_best.pth")
        
        return fold_result
    
    def evaluate_fold(self, model, val_loader, device, fold_idx):
        """
        Evaluate a single fold
        """
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, (images, labels, _) in enumerate(val_loader):
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images, labels)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        f1_macro = f1_score(all_labels, all_predictions, average='macro')
        f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
        
        fold_result = {
            'fold': fold_idx,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'predictions': all_predictions,
            'labels': all_labels
        }
        
        print(f"Fold {fold_idx + 1} Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Macro: {f1_macro:.4f}")
        print(f"  F1 Weighted: {f1_weighted:.4f}")
        
        return fold_result
    
    def train_kfold(self, dataset, model_class, config, training_fn, device):
        """
        Perform K-fold cross validation
        """
        print(f"\n=== Starting {self.n_splits}-Fold Cross Validation ===")
        
        # Create folds
        folds = self.create_folds(dataset)
        
        # Train each fold
        self.fold_results = []
        for fold_data in folds:
            fold_result = self.train_fold(fold_data, model_class, config, training_fn, device)
            self.fold_results.append(fold_result)
        
        # Calculate overall results
        self.calculate_overall_results()
        
        return self.fold_results
    
    def calculate_overall_results(self):
        """
        Calculate overall K-fold results
        """
        accuracies = [result['accuracy'] for result in self.fold_results]
        f1_macros = [result['f1_macro'] for result in self.fold_results]
        f1_weighteds = [result['f1_weighted'] for result in self.fold_results]
        
        print(f"\n=== {self.n_splits}-Fold Cross Validation Results ===")
        print(f"Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"F1 Macro: {np.mean(f1_macros):.4f} ± {np.std(f1_macros):.4f}")
        print(f"F1 Weighted: {np.mean(f1_weighteds):.4f} ± {np.std(f1_weighteds):.4f}")
        
        # Per-fold results
        print("\nPer-fold results:")
        for i, result in enumerate(self.fold_results):
            print(f"Fold {i+1}: Acc={result['accuracy']:.4f}, F1_macro={result['f1_macro']:.4f}, F1_weighted={result['f1_weighted']:.4f}")
        
        self.overall_results = {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'mean_f1_macro': np.mean(f1_macros),
            'std_f1_macro': np.std(f1_macros),
            'mean_f1_weighted': np.mean(f1_weighteds),
            'std_f1_weighted': np.std(f1_weighteds)
        }
        
        return self.overall_results
    
    def get_best_fold(self, metric='f1_macro'):
        """
        Get the best performing fold based on specified metric
        """
        if not self.fold_results:
            return None
        
        best_fold = max(self.fold_results, key=lambda x: x[metric])
        return best_fold
    
    def save_results(self, save_path):
        """
        Save K-fold results to file
        """
        import json
        
        results = {
            'overall_results': self.overall_results,
            'fold_results': [
                {
                    'fold': result['fold'],
                    'accuracy': result['accuracy'],
                    'f1_macro': result['f1_macro'],
                    'f1_weighted': result['f1_weighted']
                }
                for result in self.fold_results
            ]
        }
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {save_path}")