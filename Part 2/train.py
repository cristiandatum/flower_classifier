#import libraries
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
from PIL import Image

from collections import OrderedDict
import time
import numpy as np
import seaborn as sns
import json

'''
train.py, will train a new network on a dataset and save the model as a checkpoint. Prints out training loss, 
validation loss, and validation accuracy as the network trains.

Basic usage: python train.py data_directory

Options:
1.  Set directory to save checkpoints: python train.py data_dir --save_dir save_directory

2.  Choose architecture: python train.py data_dir --arch "vgg13"
    The training script allows users to choose from at least two different architectures available from
    torchvision.models

3.  Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
    The training script allows users to set hyperparameters for 
    , number of hidden units, 
    and training epochs.

4.  Use GPU for training: python train.py data_dir --gpu
    The training script allows users to choose training the model on a GPU
'''

'''functions order:
1. Welcome
2. input_checkpoint_dir 
3. input_model_architecture
4. input_learning_rate
5. input_hidden
6. input_epochs
7. input_gpu
8. print_models
'''

def welcome():
    print ('\n \n \n*** Welcome to Image Classifier by Cristian Alberch*** \n \n \n')


def input_checkpoint_dir():
    checkpoint_dir=input('Please type file name without extension to save trained model checkpoint: \n')
    checkpoint_dir=checkpoint_dir+".pth"
    return checkpoint_dir


def input_model_architecture():
    model_architecture=''
    model_architecture_number=input("Which model architecture would you like to choose from? \n \
    Here are the model options: \n \
    1. Densenet 121 \n \
    2. Densenet 169 \n \
    3. Oxford Visual Geometry Group (VGG16) \n")
    while True: #limits input to 1 or 2. loops until valid input is entered.
        if model_architecture_number not in ('1', '2','3'):
            print("please type '1', '2', or '3'")
            model_architecture_number=input("Which model architecture would you like to choose from? \n \
            Here are the model options: \n \
            1. Densenet 121 \n \
            2. Densenet 169 \n \
            3. VGG16 \n")
        else:
            break
    if model_architecture_number=='1':
        model_architecture='densenet121'
    if model_architecture_number=='2':
        model_architecture='densenet169'
    if model_architecture_number=='3':
        model_architecture='vgg16'
    return model_architecture

def input_learning_rate():    
    while True: #limits input from 0.001 to 0.1. loops until valid input is entered.
        try:
            learning_rate=float(input("What learning rate would you like to use to train the model? \n"))
            if learning_rate > 0.1 or learning_rate <0.001:
                raise ValueError
        except ValueError:
            print("Please enter a value between 0.001 and 0.1")
            continue
        else:
            return learning_rate
            break


def input_hidden():
    while True: #limits input to 100 to 900. loops until valid input is entered.
        try:
            hidden=int(input("How many hidden units would you like to use to train the model? \n"))
            if hidden > 900 or hidden <100:
                raise ValueError
        except ValueError:
            print("Please enter an integer between 100 and 1000 (recommended value is 1/2 of input units")
            continue
        else:
            return hidden
            break


## Function to request user number of epochs to train model
def input_epochs():
    while True: #limits input to 1 to 50. loops until valid input is entered.
        try:
            epochs=int(input("How many epochs would you like to use to train the model? \n"))
            if epochs > 51 or epochs <1:
                raise ValueError
        except ValueError:
            print("Please enter an integer between 1 and 50")
            continue
        else:
            return epochs
            break


## Function to request user to CPU or GPU to train model
def input_gpu():
    gpu=input("Do you want to use CPU or GPU to train the model? \n")
    while True: #limits input to CPU or GPU. loops until valid input is entered.
        if gpu not in ('cpu', 'CPU', 'gpu', 'GPU'):
            print("Please type 'CPU' or 'GPU':")
            gpu=input("Do you want to use CPU or GPU to train the model? \n")
        else:
            break
    if gpu=='GPU' or gpu=='gpu':
        gpu='cuda'
    else:
        gpu='cpu'
    return gpu


def user_input():
    welcome()
    checkpoint_dir=input_checkpoint_dir()
    architecture=input_model_architecture()
    learning_rate=input_learning_rate()
    hidden_units=input_hidden()
    epochs=input_epochs()
    gpu=input_gpu()
    print('you chose: "',checkpoint_dir, '" directory for checkpoint')
    print('you chose: ',architecture, 'model architecture')
    print('you chose: ',learning_rate, 'learning rate')
    print('you chose: ',hidden_units, 'hidden units for model')
    print('you chose: ', gpu,'processor training')
    print('you chose: ',epochs, 'epochs')

    return checkpoint_dir,architecture,learning_rate,hidden_units,epochs,gpu

checkpoint_dir,architecture,learning_rate,hidden_units,epochs_input,gpu = user_input()

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

def transforms_func(train_dir,valid_dir,test_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_data=datasets.ImageFolder(valid_dir, transform=valid_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    validationloader = torch.utils.data.DataLoader(valid_data, batch_size=64)

    return train_data,test_data,valid_data,trainloader,testloader,validationloader

train_data,test_data,valid_data,trainloader,testloader,validationloader=transforms_func(train_dir,valid_dir,test_dir)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Use GPU if it's available
device = torch.device("cuda" if gpu=="cuda" else "cpu")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#depending on model selection, the syntax for the number of features in the model changes. Below link provides examples:
#https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

if architecture=='densenet121':
    model = models.densenet121(pretrained=True)
    input_size = model.classifier.in_features
    
if architecture=='densenet169':
    model = models.densenet169(pretrained=True) 
    input_size = model.classifier.in_features

if architecture=='vgg16':
    model = models.vgg16(pretrained=True)
    input_size = model.classifier[0].in_features

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False
    
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model.classifier = classifier

criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

model.to(device);


# TODO: Do validation on the test set
epoch=0;epochs = epochs_input
steps = 0
running_loss = 0
print_every = 10
for epoch in range(epochs):
    print (epochs) #testing code
    print(learning_rate) #testing code
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validationloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {test_loss/len(validationloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validationloader):.3f}")
            running_loss = 0

correct = 0
total = 0
with torch.no_grad():
    
    model.eval()
    
    for inputs, labels in testloader:
        inputs = inputs.to(device)
        labels = labels.to(device)        
        
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

model.class_to_idx = train_data.class_to_idx


checkpoint = {'classifier': model.classifier,
             'class_to_idx': model.class_to_idx,
             'state_dict': model.state_dict()}

torch.save(checkpoint, checkpoint_dir)



