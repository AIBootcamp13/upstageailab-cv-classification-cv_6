import torch
from torch import optim

from trainer.evaluation import evaluation
from utils.EarlyStopping import EarlyStopping
from trainer.wandb_logger import WandbLogger


def training_loop(training_fn, model, train_dataloader, valid_dataloader, train_dataset, val_dataset, optimizer, device, num_epochs, early_stopping: EarlyStopping, logger: WandbLogger, class_names, scheduler: optim.lr_scheduler._LRScheduler=None, training_args: dict=None):
    valid_max_accuracy = -1
    start_epoch = 1

    for epoch in range(start_epoch, num_epochs):
        model, train_ret = training_fn(model, train_dataloader, train_dataset, optimizer, device, epoch, num_epochs, **training_args)
        model, valid_ret = evaluation(model, valid_dataloader, val_dataset, training_args['criterions']['focal'], device, epoch, num_epochs, class_names, logger)

        if valid_ret["valid_accuracy"] > valid_max_accuracy:
          valid_max_accuracy = valid_ret["valid_accuracy"]
        
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(valid_ret["valid_f1"])
                # scheduler.step(valid_ret["valid_loss"])
            else:
                scheduler.step(epoch - 1)  # CosineWarmRestart 등은 epoch 0번부터 시작하는거 기반
                # scheduler.step()  # 이것도 좋을 수 있다고함.
        

        # early stopping 및 check point에서 모델 저장
        early_stopping(valid_ret["valid_f1"], model)
        # early_stopping(valid_ret["valid_loss"], model)
        logger.save_model()

        print(f"Epoch [{epoch}/{num_epochs}]")
        print(f"Train Loss: {train_ret['train_loss']:.4f}, Train Accuracy: {train_ret['train_accuracy']:.4f}, Train f1: {train_ret['train_f1']}")
        print(f"Valid Loss: {valid_ret['valid_loss']:.4f}, Valid Accuracy: {valid_ret['valid_accuracy']:.4f}, Valid f1: {valid_ret['valid_f1']}")
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current LR: {current_lr:.8f}")
        
        # Epoch마다 기록
        logger.log_metrics({
            "train/loss": train_ret["train_loss"],
            "train/acc": train_ret["train_accuracy"],
            "train/f1": train_ret["train_f1"],
            "val/loss": valid_ret["valid_loss"],
            "val/acc": valid_ret["valid_accuracy"],
            "val/f1": valid_ret["valid_f1"],
            "lr": current_lr,
        }, step=epoch)
        

        if early_stopping.early_stop:
            print(f"🛑 Early stopping at epoch {epoch}")
            torch.save(model.state_dict(), "./output/last_model.pth")
            break

    logger.finish()
    
    return model, valid_max_accuracy