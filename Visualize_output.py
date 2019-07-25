'''
Created on Jun 6, 2019

@author: Shubham Sharma
'''

import numpy as np
import matplotlib.pyplot as plt
import keras
import U_Net
import VGG_
import random
from tqdm import tqdm

########################################################################
#Load the weighs and thye traning and testing npy files
x_train=np.load('./x_train.npy')
x_test=np.load('./x_test.npy')
y_test=np.load('./y_test.npy')
y_train=np.load('./y_train.npy')


width=512
height=512
depth=3
classes=2
pathtoweight1='E:/Cell_data/U-Net/'
keras.backend.set_image_data_format('channels_first')
# keras.backend.set_learning_phase(1)
net = U_Net(width, height, depth, classes,weightsPath=pathtoweight1+'Cell.h5_0183-0.9988.h5')
#########################################################################
pathtoweight2='E:/Cell_data/VGG/'


model=VGG_(weightsPath=pathtoweight2+'Cell.h5_0008-0.9979.h5')

def whitening(im):
    """
    As the images that we have are Channel_first
    """
    im = im.astype("float32")              
    for i in range(np.shape(im)[0]):                                
        im[i,:,:] = (im[i,:,:]- np.mean(im[i,:,:]))#/(np.std(im[:,:,i])+1e-9)            
    return im
def give_output_unet(x):
    '''
      Letting x to be in channel last format
    '''
    z=[]
    z.append(whitening(x))
    # Finding the probabilities of the outputs
    probs = net.predict(np.array(z))
    prediction= np.argmax(probs[0],axis=1)
    prediction = np.reshape(prediction,(512,512)) 
    prediction=prediction.astype("float32")
    return prediction

def give_output_vgg(x):
    '''
      Letting x to be in channel last format
    '''
    z=[]
    z.append(whitening(x))
    # Finding the probabilities of the outputs
    probs = model.predict(np.array(z))
    prediction= np.argmax(probs[0],axis=1)
    prediction = np.reshape(prediction,(512,512)) 
    prediction=prediction.astype("float32")
    return prediction

n=10


list=range(100)
k=random.sample(list,k=n)
plt.figure(figsize=(20, 4))
for i in tqdm(range(len(k))):
    # display original
    ax = plt.subplot(3, n, i+1)
    #As the image is in channel_first , thus converting it into channel_last
    plt.imshow(np.transpose(x_test[k[i]],[1,2,0]) )
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction VGG
    ax = plt.subplot(3, n, i + n +1)
    plt.imshow(give_output_vgg(x_test[k[i]]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # display reconstruction Unet
    ax = plt.subplot(3, n, i + 2*n +1)
    plt.imshow(give_output_unet(x_test[k[i]]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


plt.show()
