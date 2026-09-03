import torch.nn as nn
import torch.optim as optim
from model import CNN
from data_load import data_load
from torcheval.metrics import MulticlassAccuracy
import torch
from torchmetrics.classification import BinaryROC, BinaryPrecisionRecallCurve, BinaryF1Score


train_loader, val_loader, test_loader = data_load()

device = 'cuda'


def inference(saved_model_path,test_loader, model = CNN(num_classes=2)):
    metric = MulticlassAccuracy().to(device)
    roc = BinaryROC().to(device)
    f1 = BinaryF1Score().to(device)
    pr = BinaryPrecisionRecallCurve().to(device)

    model.load_state_dict(torch.load(saved_model_path,weights_only = True))
    model = model.to(device)

    model.eval()

    with torch.no_grad():
        metric.reset()
        roc.reset()
        f1.reset()
        pr.reset()

        for image_batch, labels in test_loader:
            image_batch = image_batch.to(device)
            labels = labels.to(device)
            outputs = model(image_batch)
            preds = outputs.argmax(dim=1)
            metric.update(preds,labels)
            roc.update(outputs[:,1], labels)
            pr.update(outputs[:,1], labels)
            f1.update(outputs[:,1], labels)
    acc = metric.compute()
    precision, recall, pr_threshold = pr.compute()
    f1_score = f1.compute()
    fpr, tpr, roc_threashold = roc.compute()
    return { "test_acc" :  acc, "roc": (fpr, tpr, roc_threashold), "PR_curve":(precision, recall, pr_threshold), "f1":f1_score}
        


    


if __name__ == '__main__':
    inference("../saved_models\mini_vgg_acc_ 90.00_loss_ 0.2524_epoches_300",test_loader)
    

    



