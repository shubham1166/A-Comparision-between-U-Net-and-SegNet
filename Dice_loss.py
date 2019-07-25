import numpy as np
from _23May2019_Unet.UNET_ import U_Net
from _23May2019_Unet.VGG_segmentation import VGG_
from keras import backend as K
import keras
from tqdm import tqdm
from final_post_processing.Algorithm import algorithm
# import mxnet as nd

########################################################################

width=512
height=512
depth=3
classes=2
data_shape=width*height
########################################################################
x_train=np.load('E:/Cell_data/x_train.npy')
x_test=np.load('E:/Cell_data/x_test.npy')
y_test=np.load('E:/Cell_data/y_test.npy')
y_train=np.load('E:/Cell_data/y_train.npy')
y_test = np.reshape(y_test,(len(y_test),data_shape,classes))
y_train = np.reshape(y_train,(len(y_train),data_shape,classes))
y=np.load('E:/Cell_data/label_cell.npy')
y=np.reshape(y,(y.shape[0],y.shape[1]*y.shape[2],y.shape[3]))
x_whitened=np.load('E:/Cell_data/data_cell.npy')

 
########################################################################
def dice_loss(y_true,y_pred):
    y_true=y_true.astype('float32')
    y_pred=y_pred.astype('float32')
    smooth=1
    intersection=2*sum(y_true*y_pred)
    union=sum(y_true)+sum(y_pred)
    if np.mean(union)==0 and np.mean(intersection)==0:
        dice_coef=0
    else:
        dice_coef=np.mean((intersection+smooth)/(union+smooth))
    dice_loss_=1-dice_coef
    return dice_loss_
   
   
   
   
   
   
# def give_y(net,x):
#     z=[]
#     z.append(x)
#     a=net.predict(np.array(z))
#     a=np.reshape(a,(data_shape,classes))
#     a=a.astype('uint8')
#     return a
#    
# def give_dice(x_test_whitened,y_test,net):
#     l=len(y_test)
#     dice=[]
#     for i in tqdm(range(l)):
#         y=give_y(net,x_test_whitened[i])
# #         print('giv_y is : ',y)
#         d=dice_loss(y, y_test[i])
#         dice.append(d)
#     return dice
         
 
# #############################################################################
# #UNet
# pathtoweight1='E:/Cell_data/U-Net/'
# K.set_image_data_format('channels_first')
# net = U_Net(width, height, depth, classes,weightsPath=pathtoweight1+'Cell.h5_0183-0.9988.h5')
# print('here')  
# dice_unet=give_dice(x_whitened, y, net)
# # 
# mean_dice_unet=np.mean(dice_unet)
# print('mean_dice_unet : ',mean_dice_unet)
# # #########################################################################
# #VGG
# pathtoweight2='E:/Cell_data/VGG/'
# model=VGG_(weightsPath=pathtoweight2+'Cell.h5_0008-0.9979.h5')
#       
# dice_vgg=give_dice(x_whitened, y, model)
# mean_dice_vgg=np.mean(dice_vgg)
# print('mean_dice_vgg : ',mean_dice_vgg)
#  
# ###########################################################################
# #POST-PROCESSED
# y_processed=np.load('E:/Cell_data/U-Net/Post-processing/final/images_categorical.npy')
# dice_processed=[]
# for i in tqdm(range(len(y_processed))):
#     d=dice_loss(y[i],y_processed[i])
#     dice_processed.append(d)
# mean_dice_processed=np.mean(np.array(dice_processed))
# print('mean_dice_processed : ',mean_dice_processed)
###############################################################################
#SUPER-PIXELS
y_superpixels=np.load('E:/Cell_data/Superpixels/images_categorical.npy')
dice_superpixels=[]
for i in tqdm(range(len(y_superpixels))):
#     print(i)
    d=dice_loss(y[i],y_superpixels[i])
    dice_superpixels.append(d)
mean_dice_superpixels=np.mean(np.array(dice_superpixels))
print('mean_dice_superpixels : ',mean_dice_superpixels)
print('done')  
     

      
  
  
  

