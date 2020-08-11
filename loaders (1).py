import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import requests
import urllib
import os
import pickle
from torchvision import datasets, transforms
from matplotlib import image
from matplotlib import pyplot
import time
from sklearn.model_selection import train_test_split
import re
from sklearn.metrics import f1_score
import torchvision
import random
import torch.nn.functional as F
from sklearn.metrics import precision_score
from collections import defaultdict
from collections import  Counter
import seaborn as sns


def indx_to_img_name(df):
  downloaded=np.array(df.index)
  downloaded=downloaded.astype(str)
  return np.char.add(downloaded,'.jpg')

def img_loader(input,label,batch_size,drop_last):
    
  torch.manual_seed(123)
  torch.cuda.manual_seed(123)
  np.random.seed(123)
  random.seed(123)
  torch.backends.cudnn.enabled=False
  torch.backends.cudnn.deterministic=True
  
  path='/content/gdrive/My Drive/ieeebigmm/'                                   
  transform = transforms.Compose([transforms.Resize(255),                       
                                  transforms.CenterCrop(224),                   
                                  transforms.ToTensor()])                       
  dataset = datasets.ImageFolder(path,transform=transform)

  downloaded=indx_to_img_name(input)
  
  samples=list(np.char.add('/content/gdrive/My Drive/ieeebigmm/images/',downloaded))        
  labels=list(label.values)

  data_samples=[]
  for img,label in zip(samples,labels):
    data_samples.append((img,torch.tensor(label,dtype=float)))

  dataset.samples=data_samples 
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,drop_last=drop_last)
  return dataloader

def df_split(df):
    X=df[['cleaned_text','cleaned_title']]
    y=df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y, test_size=0.33, random_state=42)

    X=X_test
    y=y_test

    X_val,X_test,y_val,y_test=train_test_split(X, y,stratify=y, test_size=0.5, random_state=42)

    return [(X_train,y_train),(X_val,y_val),(X_test,y_test)] 

def text_loader(df,column,bs,drop_last):
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True
    return torch.utils.data.DataLoader(df[column].values, batch_size=bs, shuffle=True,drop_last=drop_last)
    
                                              

