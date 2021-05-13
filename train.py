import json

import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.testing._internal.common_nn import output_size

import nltk_utils

#from PIL.TiffImagePlugin import idx
#from torch.testing._internal.distributed.rpc.examples.parameter_server_test import batch_size

from nltk_utils import tokenize, stem, bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:  #note intent from intents
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

#in our pipeline, we have to lower and stem the words

ignore_words = ['?','!','.',',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
print(all_words)
all_words = sorted(set(all_words))  #to remove duplicate element
tags = sorted(set(tags))


X_train = []
Y_train = []

for(pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    Y_train.append(label)  #sometimes we need this as one hot encoded

X_train = np.array(X_train)
Y_train = np.array(Y_train)


class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train

    #dataset[idx]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

#Hyper parameters
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0])
learnin_rate = 0.001
num_epochs = 1000


dataset = ChatDataset()
train_loader = DataLoader(dataset = dataset, batch_size= batch_size, shuffle=True, num_workers= 2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
epitimizer = torch.optim.Adam(model.parameters(), lr= learnin_rate)

for epoch in range(num_epochs):
    for(words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        #forward
        outputs = model(words)
        loss = criterion(outputs, labels)

        #backward and optimizer step
        Optimizer.zero_grad()
        loss.backward()
        Optimizer.step()

    if(epoch + 1) % 100 == 0:
        print(f'epoch {epoch + 1}/{num_epochs}, loss = {loss.item():,4f}')



