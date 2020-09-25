import numpy as np
import torch

def accuracy(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for idx in np.unique(labels_flat):
        y_hat = preds_flat[labels_flat==idx]
        y = labels_flat[labels_flat==idx]
    return (len(y_hat[y_hat==idx])/ len(y))


def evaluate(valid_dataloader):
    model.eval()
    valid_iter = iter(valid_dataloader)
    total_eval_loss = 0
    y_hat, y = [], []
    acc = 0
    for batch in valid_dataloader:
        X, Y = next(valid_iter)
        X = X.cuda()
        with torch.no_grad():
            class_pred, domain_pred = model(X, 1.)

        logits = class_pred.detach().cpu().numpy()
        label_ids = Y.cpu().numpy()
        y_hat.append(logits)
        y.append(label_ids)

    y_hat = np.concatenate(y_hat, axis=0)
    y = np.concatenate(y, axis=0)
    acc = accuracy(y_hat, y)

    return acc