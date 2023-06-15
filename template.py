#!/usr/bin/env python2.7

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from torchvision.utils import save_image


# --------------------------------------------------------------------------------
#   Neural network and helper functions; do not edit!
# --------------------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.Dropout())
        self.layer2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.Dropout())
        self.layer3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
                                    nn.Dropout())
        self.fc1 = nn.Linear(4 * 4 * 64, 256, bias=True)
        nn.init.xavier_uniform(self.fc1.weight)
        self.layer4 = nn.Sequential(self.fc1, nn.ReLU(), nn.Dropout())
        self.fc2 = nn.Linear(256, 26, bias=True)
        nn.init.xavier_uniform(self.fc2.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

# convert an image into tensor
def convert_to_tensor(path):
    img = Image.open(path)
    preprocess = transforms.Compose([transforms.Resize([28, 28]), transforms.ToTensor()])
    return preprocess(img)[None, :, :, :]

# load the machine learning model
# Note: do not edit
model = torch.load('cnn_mnist_model.pt')
model.load_state_dict(torch.load('cnn_mnist_model_state_dict.pt'))
model.eval()


# TODO: put the path to the original image
path = '3.png'
img = convert_to_tensor(path)


# prepare to run an adversarial attack
# Note: do not edit
delta = torch.zeros_like(img, requires_grad=True)
opt = optim.SGD([delta], lr=0.001)


# TODO: heuristically decide how many times you will iterate for a successful attack
for t in range(2000):

    # run forward and make a prediction
    # Note: do not edit
    img_delta = img + delta
    pred = model(img_delta)

    # TODO: Fill out the blanks with class numbers
    #       Make the model predict the image to 9
    loss = (-nn.CrossEntropyLoss()(pred, torch.LongTensor([3])) + nn.CrossEntropyLoss()(pred, torch.LongTensor([9])))
    
    
    # compute the perturbation, iteratively
    # Note: do not edit
    opt.zero_grad()
    loss.backward()
    opt.step()
    delta.data.clamp_(0, 1)

    # show the updates
    # Note: do not edit
    if (t+1) % 100 == 1:
        print('processing {}'.format(t+1))

# save the resulting adversarial example
# Note: do not edit
modified_image = img + delta
torch.save(modified_image, '3_modified.pt')

# If you want to see how the perturbed image looks like
# you can un-comment the line below and run this script again
# save_image(modified_image, '3_modified.png')
