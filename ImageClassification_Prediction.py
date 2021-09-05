# importing the libraries
import inline as inline
import pandas as pd
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
# %matplotlib inline

import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential
from torch.optim import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

test = pd.read_csv('test_ScVgIM0/test.csv')
sample_submission = pd.read_csv('sample_submission_I5njJSF.csv')

# loading test images
test_img = []
for img_name in test['id']:
    image_path = 'test_ScVgIM0/test/' + str(img_name) + '.png'
    img = imread(image_path, as_gray=True)
    img = img.astype('float32')
    test_img.append(img)

test_x = np.array(test_img)

# converting the images to 1-D
test_x = test_x/test_x.max()
test_x = test_x.reshape(-1, 28*28).astype('float32')

# number of neurons in each layer
input_num_units = 28*28
hidden_num_units = 500
output_num_units = 10

# set remaining variables
epochs = 20
learning_rate = 0.0005

# define model
model = Sequential(Linear(input_num_units, hidden_num_units),
                   ReLU(),
                   Linear(hidden_num_units, output_num_units))
# loss function
loss_fn = CrossEntropyLoss()

# define optimization algorithm
optimizer = Adam(model.parameters(), lr=learning_rate)

# getting the prediction for test images
prediction = np.argmax(model(torch.from_numpy(test_x)).data.numpy(), axis=1)

# first five rows of sample submission file
print(sample_submission.head())


# replacing the label with prediction
sample_submission['label'] = prediction
print(sample_submission.head())

# saving the file
sample_submission.to_csv('submission.csv', index=False)
