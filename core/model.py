from keras.models import Model, load_model
from keras.layers import multiply, add, Permute, Reshape, Dense, GlobalAveragePooling2D, BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Input, concatenate, Add, Concatenate
from keras import backend as K
import tensorflow as tf
import keras

from utils import get_augmented

import numpy as np

class BiONet():
    
    def __init__(self,
                 input_shape,
                 num_classes=1,
                 iterations=2,
                 multiplier=1.0,
                 activation='relu',
                 kernel_initializer='he_normal',
                 padding='same',
                 kernel_size=(3,3),
                 num_layers=4,
                 integrate=False):
        self.num_layers = num_layers
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.iterations = iterations
        self.multiplier = multiplier
        self.integrate = integrate
        
        self.filters_list = [32,64,128,256,512]
        self.bachnorm_momentum = 0.01
        
        self.conv_args = {
            'kernel_size':kernel_size,
            'activation':activation, 
            'padding':padding,
            'kernel_initializer':kernel_initializer
            }

        self.convT_args = {
            'kernel_size':kernel_size,
            'activation':activation, 
            'strides':(2,2),
            'padding':padding
            }
        
        self.maxpool_args = {
            'pool_size':(2,2),
            'strides':(2,2),
            'padding':'valid',
            }
        
    #define reusable layers
    def define_layers(self):
        
        multiplier = self.multiplier  # 这句是不是完全没有意义？？？？？？
        
        #reuse feature collections
        conv_layers = []
        deconv_layers = []
        mid_layers = []  # 这应该指的是图中最后生成语义向量的那个指针，也就是将最后一个Encoder输出内容转换为语义向量的中间层
        
        mid_layers.append(Conv2D(int(self.filters_list[self.num_layers] * multiplier),**self.conv_args))
        mid_layers.append(Conv2D(int(self.filters_list[self.num_layers] * multiplier),**self.conv_args))
        mid_layers.append(Conv2DTranspose(int(self.filters_list[self.num_layers] * multiplier), **self.convT_args))

        for l in range(self.num_layers):
            conv_layers.append(Conv2D(int(self.filters_list[l] * self.multiplier), **self.conv_args))
            conv_layers.append(Conv2D(int(self.filters_list[l] * self.multiplier), **self.conv_args))
            conv_layers.append(Conv2D(int(self.filters_list[l+1] * self.multiplier), **self.conv_args))  # 默认状态下会有四个卷积层，最终会卷到512通道

        for l in range(self.num_layers):
            deconv_layers.append(Conv2D(int(self.filters_list[self.num_layers-1-l] * self.multiplier), **self.conv_args))
            deconv_layers.append(Conv2D(int(self.filters_list[self.num_layers-1-l] * self.multiplier), **self.conv_args))
            deconv_layers.append(Conv2DTranspose(int(self.filters_list[self.num_layers-1-l] * self.multiplier), **self.convT_args))  # 最终会卷到32层，这也印证了对论文的理解，即图1中是三个反卷积（图1比此处默认的要少一层）
        
        return conv_layers, deconv_layers, mid_layers
        
    #define O-Net graph, reusable layers will be passed in
    def define_graph(self, conv_layers, mid_layers, deconv_layers):
      
        inputs = Input(self.input_shape)   

        #the first stage block is not reusable
        x = self.conv_block(inputs, self.bachnorm_momentum, int(32*self.multiplier))
        shortcut = self.conv_block(x, self.bachnorm_momentum, int(32*self.multiplier))  
        x = self.conv_block(shortcut, self.bachnorm_momentum, int(32*self.multiplier))
        x_in = MaxPooling2D(**self.maxpool_args)(x)  # 这个shortcut有啥用吗？上下文没有再使用了，应该只是顺序执行运算的啊

        back_layers = []
        collection = []
 

        for it in range(self.iterations):

            #down layers to carry forward skip connections
            down_layers = []

            for l in range(self.num_layers):  # 生成Encoder
                if l == 0:  # 如果是循环第0次，整个O形部分输入的就是刚刚处理完的x_in，否则的话应该是从decoder过来的数据
                    x = x_in

                if len(back_layers) != 0:
                    x = Concatenate()([x,back_layers[-1-l]])
                else:
                    x = Concatenate()([x,x])  # 如果back_layer不存在则自复制，否则和后面decoder传回来的数据连接
                    
                x = self.conv_block(x, self.bachnorm_momentum, conv = conv_layers[3*l])
                x = self.conv_block(x, self.bachnorm_momentum, conv = conv_layers[3*l+1])
                down_layers.append(x)  # 对应的Encoder块中的前两个CONV
                x = self.conv_block(x, self.bachnorm_momentum, conv = conv_layers[3*l+2])
                x = MaxPooling2D(**self.maxpool_args)(x)  # 对应Encoder块中的最后一个DOWN
                
            #back layers to carry backward skip connections, refresh in each inference iteration
            back_layers = []

            x = self.conv_block(x, self.bachnorm_momentum, conv=mid_layers[0])
            x = self.conv_block(x, self.bachnorm_momentum, conv=mid_layers[1])
            x = self.conv_block(x, self.bachnorm_momentum, conv=mid_layers[2])

            for l in range(self.num_layers):        
                x = concatenate([x, down_layers[-1-l]])  
                x = self.conv_block(x, self.bachnorm_momentum, conv = deconv_layers[3*l])
                x = self.conv_block(x, self.bachnorm_momentum, conv = deconv_layers[3*l+1])
                back_layers.append(x)
                
                x = self.conv_block(x, self.bachnorm_momentum, conv = deconv_layers[3*l+2])
                #integrate decoded features
                if l == self.num_layers - 1 and self.integrate:
                    collection.append(x)
        
        #to use integrate or not
        if self.integrate:
            #collection.append(shorctut)
            print(len(collection))
            x = concatenate(collection)
            
        #the last stage block is not reusable
        x = self.conv_block(x, self.bachnorm_momentum, int(32*self.multiplier))
        x = self.conv_block(x, self.bachnorm_momentum, int(32*self.multiplier))

        outputs = Conv2D(self.num_classes, kernel_size=(1,1), strides=(1,1), activation='sigmoid', padding='valid') (x)       

        model = Model(inputs=[inputs], outputs=[outputs])
        
        return model
    
    def build(self):
        conv_layers,  deconv_layers, mid_layers = self.define_layers()
        model = self.define_graph(conv_layers, mid_layers, deconv_layers)
        return model
        
    def conv_block(self,x, bachnorm_momentum,filters=None,conv=None):
        if conv is not None:
            x = conv(x)
        else:
            x = Conv2D(filters, **self.conv_args)(x)
        x = BatchNormalization(momentum=bachnorm_momentum)(x)
        return x          
