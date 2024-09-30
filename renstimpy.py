import torch
import torchvision
import torchvision.transforms as transforms
import os

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torchvision import datasets

import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import read_image

absolute_path = os.path.dirname(__file__)
train_dir = os.path.join(absolute_path, "train")
test_dir = os.path.join(absolute_path, "test")

transform = transforms.Compose(
    [
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
training_set = datasets.ImageFolder(train_dir,
                                    transform=transform)
validation_set = datasets.ImageFolder(test_dir,
                                        transform=transform)

# Create data loaders for our datasets; shuffle for training, not for validation
training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=4, shuffle=False)

# Class labels
classes = ('benign', 'malignant')

# Report split sizes
print('Training set has {} instances'.format(len(training_set)))
print('Validation set has {} instances'.format(len(validation_set)))

import matplotlib.pyplot as plt

dataiter = iter(training_loader)
images, labels = next(dataiter)

# Create a grid from the images and show them
img_grid = torchvision.utils.make_grid(images)

img = torchvision.transforms.ToPILImage()(img_grid)
img.show()

print('  '.join(classes[labels[j]] for j in range(2)))

class CancerClassifier(nn.Module):
    def __init__(self):
        super(CancerClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = CancerClassifier()

loss_fn = torch.nn.CrossEntropyLoss()

# Optimizers specified in the torch.optim package
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    print("__TOE 0.0")
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        print("__TOE 0.1")
        inputs, labels = data

        optimizer.zero_grad()

        
        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        
        print("__TOE 0.5")
        loss.backward()

        # Adjust learning weights
        
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

print("done g")


# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/cancer_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 5

best_vloss = 1_000_000.

for epoch in range(EPOCHS):

    print("__x 0")

    print('EPOCH {}:'.format(epoch_number + 1))

    print("__x 0.1")

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    
    print("__x 0.2")
            
    avg_loss = train_one_epoch(epoch_number, writer)

    print("__x 0.3")

    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()
    
    print("__x 1")

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

    print("__x 2")

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        #torch.save(model.state_dict(), model_path)
        
        model_dir = os.path.join(absolute_path, "cancer.model")
        torch.save(model.state_dict(), model_dir)

    epoch_number += 1
    
    
print("done h")




