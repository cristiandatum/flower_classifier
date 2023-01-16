#import libraries

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models

from collections import OrderedDict
import time
from PIL import Image
import numpy as np
import json

'''
predict.py, will predict flower name from an image with predict.py along with the probability of that name. 
That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

Basic usage: python predict.py data_directory


Options:
    
1.  Return top KKK most likely classes: python predict.py input checkpoint --top_k 3

2.  Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json

3.  Use GPU for inference: python predict.py input checkpoint --gpu

'''

'''functions order:
1. Welcome
2. input_image_dir
3. input_gpu
4. input_topk
5. input_json
6. print_selection
'''

# Welcoming message function
def welcome():
    print ('\n *** Welcome to Image Classifier "Classifier Predictor" by Cristian Alberch*** \n')


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

# Function to request user for name of training checkpoint
def input_model_checkpoint():
    checkpoint=input('Please type file directory and filename where checkpoint is located: \n')
    return checkpoint

# Function to request user for name of file directory
def input_image_dir():
    image_dir=input('Please type file directory where image is located: \n')
    return image_dir

# Function to request user to CPU or GPU to predict class
def input_gpu():
    gpu=input("Do you want to use CPU or GPU to predict the class? \n")
    while True: #limits input to CPU or GPU. loops until valid input is entered.
        if gpu not in ('cpu', 'CPU', 'gpu', 'GPU'):
            print("Please type 'CPU' or 'GPU':")
            gpu=input("Do you want to use CPU or GPU to predict the class? \n")
        else:
            break
    if gpu=='GPU' or gpu=='gpu':
        gpu='cuda'
    else:
        gpu='cpu'
    return gpu

## Function to request user number of top K values
def input_topk():
    while True: #limits input to 1 to 50. loops until valid input is entered.
        try:
            top_k=int(input("How many top classification values would you like to see for the model? \n"))
            if top_k > 10 or top_k <1:
                raise ValueError
        except ValueError:
            print("Please enter an integer between 1 and 10")
            continue
        else:
            return int(top_k)
            break

## Function to request user json file for dictionary mapping
def input_json():
    json_dir=input('Please type file directory and filename where json dictionary mapping is located: \n')
    with open(json_dir, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name
    
def user_input():
    welcome()
    architecture=input_model_architecture()
    checkpoint=input_model_checkpoint()
    image_path=input_image_dir()
    dictionary=input_json()
    gpu=input_gpu()
    topk=input_topk()

    print('you chose: "',architecture, '" model type')
    print('you chose: "',checkpoint, '" checkpoint file name')
    print('you chose: "',image_path, '" directory for checkpoint')
    print('you chose: ', gpu,'processor predicting')
    print('you chose: ',topk, 'top classifier predictors')
    return architecture,checkpoint,image_path,dictionary,gpu,topk

architecture,checkpoint,image_path,dictionary,gpu,topk = user_input()

# Use GPU if it's selected
device = torch.device("cuda" if gpu=="cuda" else "cpu")

#run predict.py
#A function that loads a checkpoint and rebuilds the model

def load_model_checkpoint(architecture,checkpoint_dir):
    checkpoint = torch.load(checkpoint_dir)

    if architecture=='densenet121':
        model = models.densenet121(pretrained=True)
        input_size = model.classifier.in_features
    
    if architecture=='densenet169':
        model = models.densenet169(pretrained=True) 
        input_size = model.classifier.in_features

    if architecture=='vgg16':
        model = models.vgg16(pretrained=True)
        input_size = model.classifier[0].in_features
   
    for param in model.parameters(): 
        param.requires_grad = False    
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model

model = load_model_checkpoint(architecture,checkpoint)

print(model.classifier)

def process_image(image_path):
    img = Image.open(image_path,'r')
    width=img.size[0]
    height=img.size[1]
    if width>height:
        size=height*2,256
    else:
        size=256,width*2
    img.thumbnail(size)   
    x_width=img.size[0]
    y_height=img.size[1]
    left=0; top=0
    right=224; bottom=224
    new_side=224
    if x_width>y_height:  #width > height (only need to change the left and right sides)
        left=(x_width-new_side)/2
        right=x_width-(x_width-new_side)/2
        top=16
        bottom=240
    else:  #height > width or square (only need to change the top and bottom sides)
        left=16
        right=240
        top=(y_height-new_side)/2
        bottom=y_height-(y_height-new_side)/2
    img = img.crop((left, top, right, bottom))
    np_image=np.array(img)/255 
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image=(np_image-mean)/std    
    np_image = np.transpose(np_image, (2, 0, 1))
    return np_image

def predict(image_path, model, topk,dictionary):    
    np_img = process_image(image_path)
    image_tensor = torch.from_numpy(np_img).type(torch.FloatTensor)
    model.eval();
    model_input = image_tensor.unsqueeze(0)
    probs = torch.exp(model.forward(model_input.to(device)))
    top_probs, top_labels = probs.topk(topk)
    top_probs=top_probs.cpu()
    top_labels=top_labels.cpu()
    top_probs = top_probs.detach().numpy().tolist()[0] 
    top_labels = top_labels.detach().numpy().tolist()[0]
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[label] for label in top_labels]
    top_flowers=[]
    for label in top_labels: top_flowers.append(dictionary[label])            
    return top_probs, top_labels, top_flowers

model.cuda() #change depending on which selection was made

top_probs,top_labels,top_flowers=predict(image_path, model, topk,dictionary)

for i in range(topk):
    print("top", i+1, "out of",topk)
    print("flower:",top_flowers[i])
    print("probability:",round(top_probs[i],4)," \n")
 