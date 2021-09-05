# importing the libraries
import inline as inline
import pandas as pd
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
# %matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# loading dataset
train = pd.read_csv('train_LbELtWX/train.csv')
test = pd.read_csv('test_ScVgIM0/test.csv')

sample_submission = pd.read_csv('sample_submission_I5njJSF.csv')

print(train.head())

# random number generator
seed = 128
rng = np.random.RandomState(seed)

# print an image
img_name = rng.choice(train['id'])

filepath = 'train_LbELtWX/train/' + str(img_name) + '.png'

img = imread(filepath, as_gray=True)
img = img.astype('float32')

plt.figure(figsize=(5,5))
plt.imshow(img, cmap='gray')
plt.show()

# loading training images
train_img = []
for img_name in train['id']:
    image_path = 'train_LbELtWX/train/' + str(img_name) + '.png'
    img = imread(image_path, as_gray=True)
    img = img.astype('float32')
    train_img.append(img)

train_x = np.array(train_img)
train_x.shape

train_x = train_x/train_x.max()
train_x = train_x.reshape(-1, 28*28).astype('float32')
train_x.shape

train_y = train['label'].values

print(train_y)


