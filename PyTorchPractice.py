import numpy as np
import torch

# initializing a numpy array
a = np.array(1)

# initializing a tensor
b = torch.tensor(1)

print(a)
print(b)
print(type(a), type(b))

# Numpy initializing two arrays
a = np.array(2)
b = np.array(1)
print(a,b)

# addition
print(a+b)

# subtraction
print(b-a)

# multiplication
print(a*b)

# division
print(a/b)

# Torch initializing two tensors
a = torch.tensor(2)
b = torch.tensor(1)
print(a,b)

# addition
print(a+b)

# subtraction
print(b-a)

# multiplication
print(a*b)

# division
print(a/b)

# matrix of zeros
a = np.zeros((3,3))
print(a)
print(a.shape)

a = torch.zeros((3,3))
print(a)
print(a.shape)


