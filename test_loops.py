import torch
import numpy as np
from tqdm import tqdm

def test_loop_coronal_npy(dataloader, model, loss, output_npy, device='cuda', label=True):
    num = len(dataloader.dataset)
    size = len(dataloader.dataset)
    batches = len(dataloader)
    bar = tqdm(dataloader)
    test_loss, test_metric = 0, 0
    if label:
        x, y = dataloader.dataset[0]
    else:
        x = dataloader.dataset[0]
    w, h = x.shape[1], x.shape[2]
    dataset = np.zeros((w, num, h), dtype=np.float32)

    i = 0
    with torch.no_grad():
        if label:
            for X, y in bar:
                X = X.to(device, dtype=torch.float)
                pred = model(X)
                dataset[:,i,:] = pred.cpu().numpy()
                label = y.to(device, dtype=torch.float)
                cur_loss = loss(pred, label)
                test_loss += cur_loss / batches
                bar.set_description(f"test loss: {cur_loss:>7f}")
                i+=1
            print(f"Avg loss on whole image: {test_loss:>8f} \n")

        else:
            for X in bar:
                X = X.to(device, dtype=torch.float)
                pred = model(X)
                dataset[:,i,:] = pred.cpu().numpy()
                i+=1
    np.save(output_npy, dataset)
    return dataset

def test_loop_saggital_npy(dataloader, model, loss, output_npy, device='cuda', label=True):
    num = len(dataloader.dataset)
    size = len(dataloader.dataset)
    batches = len(dataloader)
    bar = tqdm(dataloader)
    test_loss, test_metric = 0, 0

    if label:
        x, y = dataloader.dataset[0]
    else:
        x = dataloader.dataset[0]
    w, h = x.shape[1], x.shape[2]
    dataset = np.zeros((w, h, num), dtype=np.float32)

    i = 0
    with torch.no_grad():
        if label:
            for X, y in bar:
                X = X.to(device, dtype=torch.float)
                pred = model(X)
                dataset[:,:,i] = pred.cpu().numpy() #/ dataloader.dataset.factor
                label = y.to(device, dtype=torch.float)
                cur_loss = loss(pred, label)
                test_loss += cur_loss / batches
                bar.set_description(f"test loss: {cur_loss:>7f}")
                i+=1
            print(f"Avg loss on whole image: {test_loss:>8f} \n")

        else:
            for X in bar:
                X = X.to(device, dtype=torch.float)
                pred = model(X)
                dataset[:, :, i] = pred.cpu().numpy()
                i += 1
    np.save(output_npy, dataset)
    return dataset