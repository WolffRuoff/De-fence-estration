import torch
import numpy as np
import gc
import Model.trainer as trainer

# forward pass for inference and evaluation
def get_inference_output(model, X_in, device):
    model.eval()
    X_in = X_in.to(device)
    with torch.no_grad():
        output = model(X_in.float())
    return output

def evaluate_model(model, loader_val, loss_fn, device):
    acc_val = []
    loss_val = []
    recall_val = []
    precision_val = []
    err_x, err_y, err_pred = np.array([]), np.array([]), np.array([])
    val_batch_num, val_num_batches = 0, len(loader_val)
    # set model to eval mode when evaluating on validation set
    model.eval()
    for X_val, y_val in loader_val:
        # only take the first color channel of mask
        y_val = y_val[:, 0, :, :]
        # flatten the mask image
        y_val = y_val.reshape(-1, y_val.shape[-2] * y_val.shape[-1])
        X_val, y_val = X_val.to(device), y_val.to(device)
        
        with torch.no_grad():
            # output here is logit (before passing through sigmoid)
            output = get_inference_output(model, X_val, device)
            # class=1 if logit > 0 is equivalent to class=1 if sigmoid(logit) > 0.5
            predictions = torch.where(output > 0, 1, 0)
            batch_loss = loss_fn(output, y_val.float())
            
        batch_acc = trainer.get_acc(predictions, y_val)
        batch_recall = trainer.get_recall(predictions, y_val)
        batch_precision = trainer.get_precision(predictions, y_val)
        
        acc_val.append(batch_acc.item())
        loss_val.append(batch_loss.item())
        recall_val.append(batch_recall.item())
        precision_val.append(batch_precision.item())
        
        print('evaluating batch %d/%d'%(val_batch_num+1, val_num_batches), end='\r')
        val_batch_num += 1  
        
    del X_val
    del y_val
    torch.cuda.empty_cache()
    gc.collect()    

    # get validation metrics for this epoch
    total_acc_val = np.mean(acc_val)
    total_loss_val = np.mean(loss_val)
    total_recall_val = np.mean(recall_val)
    total_precision_val = np.mean(precision_val)
    # set model back to training mode after finishing evaluation
    model.train()
    
    return (total_loss_val, total_acc_val, total_recall_val, total_precision_val)