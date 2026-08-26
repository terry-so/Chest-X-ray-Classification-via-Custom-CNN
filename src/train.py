import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from model import CNN
from torchvision import models
import torch
from torcheval.metrics import MulticlassAccuracy
import copy


def training_loop(train_loader, val_loader, model = CNN(num_classes=2), lr = 1e-3, num_epoch = 300, patience = 30, factor=0.1):
    device = "cuda"
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor = factor, patience=10)
    best_loss = float("inf")
    best_acc = None
    metric = MulticlassAccuracy()
    val_loss_list = []
    val_acc_list = []
    train_loss_list = []

    for epoch in range(num_epoch):
        model.train()
        metric.reset()
        epoch_train_loss = 0
        for image_batch, labels in train_loader:

            image_batch = image_batch.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(image_batch)
            
            train_loss = criterion(outputs, labels)
            train_loss.backward()
            optimizer.step()
            epoch_train_loss += train_loss.item()
        
        epoch_avg_train_loss = epoch_train_loss/len(train_loader)
        
        model.eval()
        with torch.no_grad():
            epoch_val_loss = 0
            
            for image_batch, labels in val_loader:
                
                image_batch = image_batch.to(device)
                labels = labels.to(device)

                outputs = model(image_batch)
                val_loss = criterion(outputs, labels)
                epoch_val_loss += val_loss.item()
                preds = outputs.argmax(dim=1)
                metric.update(preds, labels)

            epoch_avg_val_loss = epoch_val_loss/len(val_loader)
            scheduler.step(epoch_avg_val_loss)
            
            acc = metric.compute()
            if epoch_avg_val_loss < best_loss:
                best_loss = epoch_avg_val_loss
                best_acc = acc.item()
                best_model = copy.deepcopy(model.state_dict())
                no_improvement_count = 0
                
            else:
                no_improvement_count += 1
        
                
                
        


            print(f"Epoch {epoch}  Average Training Loss: {epoch_avg_train_loss: .4f}, Average Validation Loss: {epoch_avg_val_loss: .4f}, Accuracy: {acc: .4f}, Best Validation Loss: {best_loss: .4f}")

            train_loss_list.append(epoch_avg_train_loss)
            val_loss_list.append(epoch_avg_val_loss)
            val_acc_list.append(acc)
            print(optimizer.param_groups[0]['lr'])

            if no_improvement_count == patience:
                print(f"Best Average Validation Loss: {best_loss:.4f}, Saving Model..")
                torch.save(best_model, f"../saved_models/mini_vgg_acc_{best_acc*100:.2f}_loss_{best_loss:.4f}_lr_{lr}_epoches_{epoch}_patience_{patience}")
                print('Model Saved')
                
                return train_loss_list, val_loss_list, val_acc_list


    print(f"Best Average Validation Loss: {best_loss:.4f}, Saving Model..")
    torch.save(best_model, f"../saved_models/mini_vgg_acc_{best_acc*100:.2f}_loss_{best_loss:.4f}_lr_{lr}_epoches_{num_epoch}_patience_{patience}")
    print('Model Saved')

    return train_loss_list, val_loss_list, val_acc_list


   

        
    

    


if __name__ == "__main__":
    from data_load import train_loader, val_loader, test_loader
    
   
    train_loss_list, val_loss_list, val_acc_list = training_loop(train_loader,val_loader)
    print(train_loss_list)
    print(val_loss_list)
    print(val_acc_list)
    

    

