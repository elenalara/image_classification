# Model class: training and testing, analysis in tensorboard, predicting

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from monai.networks.nets import DenseNet201, EfficientNetBN
import monai.transforms as transforms
from monai.metrics import ROCAUCMetric
from monai.data import decollate_batch
from sklearn.metrics import classification_report, confusion_matrix

import tools
import dataload
import settings

class MyNeuralNetwork:

    def train(self, train_ds, test_ds, device, class_names, batch_size, max_epochs, writer, progress, pretrained, finetune, type_net, model_name, lr, save_model=True):
        # Defining train and test sets
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=10)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, num_workers=10)
        
        # Neural network, loss function, optimizer
        loss_function = torch.nn.CrossEntropyLoss()
        auc_metric = ROCAUCMetric()
        if pretrained:
            if type_net == 'dense':
                net = DenseNet201(spatial_dims=2, in_channels=3, out_channels=len(class_names), pretrained=True)
            else:
                net = EfficientNetBN("efficientnet-b0", pretrained=True, spatial_dims=2, in_channels=3, num_classes=len(class_names))
            if not finetune:
                for param in net.parameters():
                    param.requires_grad = False
            if type_net == 'dense':
                num_ftrs = net.class_layers.out.in_features
                net.class_layers.out = torch.nn.Linear(num_ftrs, len(class_names)) #this one has requires_grad=True
            else:
                num_ftrs = net._fc.in_features
                net._fc = torch.nn.Linear(num_ftrs, len(class_names)) #this one has requires_grad=True
            net = net.to(device)
            if not finetune:
                if type_net == 'dense':
                    optimizer = torch.optim.Adam(net.class_layers.out.parameters(), lr=lr)
                else:
                    optimizer = torch.optim.Adam(net._fc.parameters(), lr=lr)
            else:
                optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        else: # https://keras.io/api/applications/
            if type_net == 'mine':
                class Net(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.conv1 = nn.Conv2d(3, 6, 5)
                        self.pool = nn.MaxPool2d(2, 2)
                        self.conv2 = nn.Conv2d(6, 16, 5)
                        self.fc1 = nn.Linear(16 * 5 * 5, 120)
                        self.fc2 = nn.Linear(120, 84)
                        self.fc3 = nn.Linear(84, 10)

                    def forward(self, x):
                        x = self.pool(F.relu(self.conv1(x)))
                        x = self.pool(F.relu(self.conv2(x)))
                        x = torch.flatten(x, 1) # flatten all dimensions except batch
                        x = F.relu(self.fc1(x))
                        x = F.relu(self.fc2(x))
                        x = self.fc3(x)
                        return x
                net = Net().to(device)
            elif type_net == 'dense':
                net = DenseNet201(spatial_dims=2, in_channels=3, out_channels=len(class_names)).to(device)
            else:
                net = EfficientNetBN("efficientnet-b0", pretrained=False, spatial_dims=2, in_channels=3, num_classes=len(class_names)).to(device)
            optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1) #Decays the learning rate of each parameter group by 0.1 every 40 epochs
        
        # Tensorboard---------------------------------------------------------------------------
        dataiter = iter(train_loader) # get a random batch of training images of size batch_size
        images = dataiter.next()[0].to(device)
        # write to tensorboard
        writer.add_graph(net, images) # visualize the model we built -> “Graphs”
        writer.flush()

        print("Training")
        epoch_loss = 0.0
        best_metric = -1.0
        best_epoch = -1
        num_each = {classname: 0 for classname in class_names}
        for epoch in range(max_epochs):
            print("-" * 20)
            print(f"epoch {epoch + 1}/{max_epochs}")
            net.train()
            train_prob = torch.tensor([], dtype=torch.float32, device=device)
            train_true = torch.tensor([], dtype=torch.long, device=device)
            for step, batch_data in enumerate(train_loader, 0):
                inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
                for label in labels:
                    num_each[class_names[label]] += 1
                optimizer.zero_grad()
                outputs = net(inputs)
                train_prob = torch.cat([train_prob, outputs])
                train_true = torch.cat([train_true, labels])
                loss = loss_function(outputs, labels)
                loss.backward() # backpropagation
                optimizer.step()
                step_len = len(train_ds) // train_loader.batch_size
                if progress:
                    print(f"{step}/{step_len}, train_loss: {loss.item():.4f}")
                epoch_loss += loss.item()
            epoch_loss /= step + 1
            writer.add_scalar('Loss', epoch_loss, epoch) # "Scalars"
            train_acc_value = torch.eq(train_prob.argmax(dim=1), train_true)
            train_acc_metric = train_acc_value.sum().item() / len(train_acc_value)
            writer.add_scalars('Accuracy', {'train': train_acc_metric}, epoch) # "Scalars"
            writer.flush()

            print("Validation")
            net.eval()
            with torch.no_grad():
                val_prob = torch.tensor([], dtype=torch.float32, device=device)
                val_true = torch.tensor([], dtype=torch.long, device=device)
                for val_data in test_loader:
                    val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                    val_prob = torch.cat([val_prob, net(val_images)])
                    val_true = torch.cat([val_true, val_labels])
                #Accuracy
                acc_value = torch.eq(val_prob.argmax(dim=1), val_true)
                acc_metric = acc_value.sum().item() / len(acc_value)
                writer.add_scalars('Accuracy', {'validation': acc_metric}, epoch) # "Scalars"
                writer.flush()
                if save_model:
                    if acc_metric > best_metric:
                        best_metric = acc_metric
                        best_metric_epoch = epoch + 1
                        torch.save(net.state_dict(), settings.MODELS_PATH / model_name)
                        print("saved new best metric model")
                
                print(f"Epoch: {epoch + 1} lr: {optimizer.param_groups[0]['lr']} average loss: {epoch_loss:.4f} accuracy: {acc_metric:.4f}")
                epoch_loss = 0.0
            
            scheduler.step()

        print("Testing")
        net.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            class_images = []
            class_probs = []
            class_label = []
            for test_data in test_loader:
                images, labels = test_data[0].to(device), test_data[1].to(device)
                probs_batch = net(images)
                pred = probs_batch.argmax(dim=1)
                for i in range(len(pred)):
                    y_true.append(labels[i].item())
                    y_pred.append(pred[i].item())
                class_probs_batch = [F.softmax(i, dim=0) for i in probs_batch]
                class_images.append(images)
                class_probs.append(class_probs_batch)
                class_label.append(labels)
        test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
        test_label = torch.cat(class_label)
        test_images = torch.cat(class_images)

        writer.add_figure('tests', tools.plot_predictions_true(class_names, test_probs, test_label, test_images, nrows=3, ncols=3))
        writer.add_text('acc', classification_report(y_true, y_pred, labels=range(len(class_names)), target_names=class_names, digits=6))
        writer.flush()
        print(classification_report(y_true, y_pred, labels=range(len(class_names)), target_names=class_names, digits=6))
        
        # plot confusion matrix
        cmat = confusion_matrix(y_true, y_pred)
        writer.add_figure('Confusion matrix', tools.plot_confusion_matrix(cmat, class_names=class_names))

        # plot all the pr curves
        for i in range(len(class_names)):
            tensorboard_truth = test_label == i
            tensorboard_probs = test_probs[:, i]
            writer.add_pr_curve(class_names[i], tensorboard_truth, tensorboard_probs)
            writer.flush()
        
        self.net = net
        return num_each

    @classmethod
    def load(self, filename, device, class_names):
        """ Load model from file """
        model = DenseNet121(spatial_dims=2, in_channels=3, out_channels=len(class_names)).to(device)
        model.load_state_dict(torch.load(filename))
        self.net = model

    def save(self, filename):
        """ Save a model in a file """
        torch.save(self.net.state_dict(), filename)
