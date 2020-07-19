import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from neural.dataset import CustomDataset
from neural.nn_models import Net
from neural.nn_utils import to_one_hot, dataset_accuracy

data_path = 'data/corona_tested_individuals_ver_0043.csv'
n_epochs = 30
cuda_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = Net()
net.to(cuda_device)
if torch.cuda.is_available():
    print("using cuda")

optimizer = optim.Adam(params=net.parameters(), lr=1e-3)
# loss_f = nn.BCELoss()
loss_f = nn.BCEWithLogitsLoss()

df = pd.read_csv(data_path)
train, test, _, _ = train_test_split(df, df.corona_result, test_size=0.33,
                                     random_state=42)

training_ds = CustomDataset(train)
training_dataloader = DataLoader(training_ds, batch_size=1000)
tranining_tensor = torch.stack(
    [training_ds[i][0] for i in range(len(training_ds))])
training_labels = training_ds.labels.data.numpy()

validation_ds = CustomDataset(test)
validation_dataloader = DataLoader(validation_ds, batch_size=1000)
validation_tensor = torch.stack(
    [validation_ds[i][0] for i in range(len(validation_ds))])
validation_labels = validation_ds.labels.data.numpy()

validation_loss_min = np.Inf

loss_vs_epoch = []
accuracy_vs_epoch = []
recall_vs_epoch = []
precision_vs_epoch = []

for epoch in tqdm(range(n_epochs)):
    net.train()
    for x, y in training_dataloader:
        if torch.cuda.is_available():
            x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        pred = net(x)
        training_loss = loss_f(pred, to_one_hot(y))
        training_loss.backward()
        optimizer.step()
    training_accuracy, training_recall, training_precision, _ = dataset_accuracy(
        tranining_tensor, training_labels, net, loss_f, cuda_device)

    net.eval()
    for x, y in validation_dataloader:
        if torch.cuda.is_available():
            x, y = x.cuda(), y.cuda()
        pred = net(x)
        validation_loss = loss_f(pred, to_one_hot(y))
    validation_accuracy, validation_recall, validation_precision, _ = dataset_accuracy(
        validation_tensor, validation_labels, net, loss_f, cuda_device)

    loss_vs_epoch.append([training_loss.item(), validation_loss.item()])
    accuracy_vs_epoch.append([training_accuracy, validation_accuracy])
    recall_vs_epoch.append([training_recall, validation_recall])
    precision_vs_epoch.append([training_precision, validation_precision])

    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, training_loss, validation_loss))

    if validation_loss <= validation_loss_min:
        print(
            'Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                validation_loss_min, validation_loss))
        torch.save(net.state_dict(), 'corn_model2.pt')
        validation_loss_min = validation_loss
