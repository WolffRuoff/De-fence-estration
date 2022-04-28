import torch
import numpy as np
import os
import gc
import torch.nn as nn
import evaluator


def get_acc(predictions, labels):
    # labels.shape[0]*labels.shape[1] is height * width, since we are doing pixel-level classification
    return torch.sum(predictions == labels)/(labels.shape[0]*labels.shape[1])


def get_recall(predictions, labels):
    # True Positive / all positive
    return torch.sum((predictions == labels) * (labels == 1))/torch.sum(labels == 1)


def get_precision(predictions, labels):
    # True Positive / preditcted positive
    return torch.sum((predictions == labels) * (labels == 1))/torch.sum(predictions == 1)


# This function is used to train the model
# Reference: pytorch official tutorial
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
def train(model, loader_train, loader_val, lr=1e-4, num_epochs=10, device='cpu', patience=5, evaluation_interval=None, pos_weight=None):
    # initialize lists to store logs of the validation loss and validation accuracy
    val_loss_hist = []
    val_acc_hist = []

    # initialize optimizer with specified learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # From documentation: This loss combines a Sigmoid layer and the BCELoss in one single class.
    if pos_weight:
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.Tensor(pos_weight).to(device))
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    # early-stopping parameters
    param_hist = []
    best_n_loss = None
    current_patience = patience
    # the number of training steps between evaluations
    stop_early = False

    for e in range(num_epochs):
        # set model to training mode
        model.train()
        # current batch index
        batch_num, num_batches = 0, len(loader_train)
        batch_acc_train = []
        batch_loss_train = []
        batch_recall_train = []
        batch_precision_train = []

        # Training pass
        for X_batch, y_batch in loader_train:
            
            # only take the first color channel of mask
            y_batch = y_batch[:, 0, :, :]
            # flatten the mask image
            y_batch = y_batch.reshape(-1, y_batch.shape[-2] * y_batch.shape[-1])

            # torch tensor can be loaded to GPU, when applicable
            X_batch, y_batch = X_batch.float().to(device), y_batch.to(device)

            # reset gradients for the optimizer, need to be done each training step
            optimizer.zero_grad()

            # output here is logit (before passing through sigmoid)
            output = model(X_batch)
            # class=1 if logit > 0 is equivalent to class=1 if sigmoid(logit) > 0.5
            predictions = torch.where(output > 0, 1, 0)
            # loss_fn here is BCEWithLogitsLoss, which again includes the sigmoid layer
            batch_loss = loss_fn(output, y_batch.float())
            # compute the gradients and take optimization step
            batch_loss.backward()
            optimizer.step()

            batch_acc = get_acc(predictions, y_batch)
            batch_recall = get_recall(predictions, y_batch)
            batch_precision = get_precision(predictions, y_batch)

            batch_acc_train.append(batch_acc.item())
            batch_loss_train.append(batch_loss.item())
            batch_recall_train.append(batch_recall.item())
            batch_precision_train.append(batch_precision.item())

            # running average
            avg_train_acc = np.mean(batch_acc_train)
            avg_train_loss = np.mean(batch_loss_train)
            avg_train_recall = np.mean(batch_recall_train)
            avg_train_precision = np.mean(batch_precision_train)

            print('Training epoch %d batch %d/%d, train loss = %f, train acc = %f, recall = %f, precision = %f'
                  % (e+1, batch_num+1, num_batches, avg_train_loss,
                     avg_train_acc, avg_train_recall, avg_train_precision), end='\r')

            batch_num += 1

            if batch_num % 20 == 0:
                del X_batch
                del y_batch
                torch.cuda.empty_cache()
                gc.collect()

            if not evaluation_interval:
                evaluation_interval = num_batches//2

            # evaluate on validation set every 100 epochs, invoke early-stopping as needed (with patience)
            if batch_num % evaluation_interval == 0 or batch_num == num_batches:

                # evaluate the model
                print()
                total_loss_val, total_acc_val, total_recall_val, total_precision_val = evaluator.evaluate_model(
                    model, loader_val, loss_fn, device)

                print('validation metrics at epoch %d batch %d: val loss = %f, val acc = %f, val recall = %f, val precision = %f'
                      % (e+1, batch_num, total_loss_val, total_acc_val, total_recall_val, total_precision_val))

                val_loss_hist.append(total_loss_val)
                val_acc_hist.append(total_acc_val)

                # early stopping with patience
                save_path = 'epoch_%d_batch_%d.model' % (e, batch_num)
                torch.save(model.state_dict(), save_path)
                param_hist.append(save_path)
                # only need to keep weights needed for earlystopping
                if len(param_hist) > patience+1:
                    del_path = param_hist.pop(0)
                    os.remove(del_path)  # delete unnecessary state dicts
                if best_n_loss and total_loss_val >= best_n_loss:
                    current_patience -= 1
                    print('current_patience = %d' % current_patience)
                    if current_patience == 0:
                        print('\nstopping early after no validation accuracy improvement in %d steps'
                              % (patience * evaluation_interval))
                        best_weights_path = param_hist[-(patience+1)]
                        # restore to last best weights when stopping early
                        model.load_state_dict(torch.load(best_weights_path))
                        stop_early = True
                        break

                # if performance improves, reset patience and best accuracy
                else:
                    current_patience = patience
                    best_n_loss = total_loss_val

        if stop_early:
            break

        # get epoch-wide training metrics
        epoch_loss_train = np.mean(batch_loss_train)
        epoch_acc_train = np.mean(batch_acc_train)

        print('='*80+'\nEpoch %d/%d train loss = %f, train acc = %f, val loss = %f, val acc = %f'
              % (e+1, num_epochs, epoch_loss_train, epoch_acc_train, total_loss_val, total_acc_val))

    if device == 'cuda':
        torch.cuda.empty_cache()  # free gpu memory if loaded to cuda
    else:
        gc.collect()  # free memory if not using cuda

    # remove cached weights after stopping and loading best weights (if applicable)
    cached_weight_paths = [f for f in os.listdir(
        '.') if ('epoch' in f and '.model' in f)]
    for p in cached_weight_paths:
        os.remove(p)

    return (val_loss_hist, val_acc_hist)
