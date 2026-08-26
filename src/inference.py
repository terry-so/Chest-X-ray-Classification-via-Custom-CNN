import torch.nn as nn
import torch.optim as optim
from model import CNN
from data_load import data_load
from torcheval.metrics import MulticlassAccuracy
import torch

train_loader, val_loader, test_loader = data_load()
metric = MulticlassAccuracy()
device = 'cuda'


def inference(saved_model_path,test_loader, model = CNN(num_classes=2)):
    model.load_state_dict(torch.load(saved_model_path,weights_only = True))
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        metric.reset()
        for image_batch, labels in test_loader:
            image_batch = image_batch.to(device)
            labels = labels.to(device)
            outputs = model(image_batch)
            preds = outputs.argmax(dim=1)
            metric.update(preds,labels)
    acc = metric.compute()
    print(f"Test Accuracy: {acc: .2f}")
        


    


if __name__ == '__main__':
    inference("../saved_models\mini_vgg_acc_ 90.00_loss_ 0.2524_epoches_300",test_loader)
    

    



