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

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader,Dataset
from PIL import Image

absolute_path = os.path.dirname(__file__)

class CancerClassifier(nn.Module):
    def __init__(self):
        super(CancerClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model_dir = os.path.join(absolute_path, "cancer.model")
model = CancerClassifier()
model.load_state_dict(torch.load(model_dir, weights_only=True), strict=False)

image_dir = os.path.join(absolute_path, "melanoma_224.png")

transform = transforms.Compose(
    [transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def pre_image(image_path,model):

   img = Image.open(image_path)
   
   img.show()
   
   img = transform(img)
   
   with torch.no_grad():
      model.eval()
      output = model(img)
      index = output.data.cpu().numpy().argmax()
      print("Index = ", index)
      classes = ('benign', 'malignant')
      class_name = classes[index]
      return class_name
      
predict_class = pre_image(image_dir, model)
print(predict_class)
