from keras.models import Model, load_model
from keras.layers import multiply, add, Permute, Reshape, Dense, GlobalAveragePooling2D, BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Input, concatenate, Add, Concatenate
from keras import backend as K
import tensorflow as tf
import keras

import numpy as np

class BiONet(object):
    
    def __init__(self,
                 input_shape,
                 num_classes=1,
                 iterations=2, 
                 multiplier=1.0,
                 num_layers=4,
                 integrate=False):

        activation='relu'
        kernel_initializer='he_normal'
        padding='same'
        kernel_size=(3,3)
        self.num_layers = num_layers
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.iterations = iterations
        self.multiplier = multiplier
        self.integrate = integrate
        
        self.filters_list = [int(32 * (2 ** i) * self.multiplier) for i in range(self.num_layers + 1)]
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
        
        #reuse feature collections
        conv_layers = []
        deconv_layers = []
        mid_layers = []
        
        mid_layers.append(Conv2D(self.filters_list[self.num_layers],**self.conv_args))
        mid_layers.append(Conv2D(self.filters_list[self.num_layers],**self.conv_args))
        mid_layers.append(Conv2DTranspose(self.filters_list[self.num_layers], **self.convT_args))

        for l in range(self.num_layers):
            conv_layers.append(Conv2D(self.filters_list[l], **self.conv_args))
            conv_layers.append(Conv2D(self.filters_list[l], **self.conv_args))
            conv_layers.append(Conv2D(self.filters_list[l+1], **self.conv_args))

        for l in range(self.num_layers):
            deconv_layers.append(Conv2D(self.filters_list[self.num_layers-1-l], **self.conv_args))
            deconv_layers.append(Conv2D(self.filters_list[self.num_layers-1-l], **self.conv_args))
            deconv_layers.append(Conv2DTranspose(self.filters_list[self.num_layers-1-l], **self.convT_args))
        
        return conv_layers, deconv_layers, mid_layers
        
    #define BiO-Net graph, reusable layers will be passed in
    def define_graph(self, conv_layers, mid_layers, deconv_layers):
      
        inputs = Input(self.input_shape)   

        #the first stage block is not reusable
        x = self.conv_block(inputs, self.bachnorm_momentum, int(32*self.multiplier))
        shortcut = self.conv_block(x, self.bachnorm_momentum, int(32*self.multiplier))  
        x = self.conv_block(shortcut, self.bachnorm_momentum, int(32*self.multiplier))
        x_in = MaxPooling2D(**self.maxpool_args)(x)

        back_layers = []
        collection = []
 

        for it in range(self.iterations):

            #down layers to carry forward skip connections
            down_layers = []

            # encoding stage
            for l in range(self.num_layers):
                if l == 0:
                    x = x_in

                if len(back_layers) != 0:
                    x = Concatenate()([x,back_layers[-1-l]]) # backward skip connection
                else:
                    x = Concatenate()([x,x])  # self concatenation if there is no backward connections
                    
                x = self.conv_block(x, self.bachnorm_momentum, conv = conv_layers[3*l])
                x = self.conv_block(x, self.bachnorm_momentum, conv = conv_layers[3*l+1])
                down_layers.append(x)
                x = self.conv_block(x, self.bachnorm_momentum, conv = conv_layers[3*l+2])
                x = MaxPooling2D(**self.maxpool_args)(x)
                
            #back layers to carry backward skip connections, refresh in each inference iteration
            back_layers = []

            # middle stage
            x = self.conv_block(x, self.bachnorm_momentum, conv=mid_layers[0])
            x = self.conv_block(x, self.bachnorm_momentum, conv=mid_layers[1])
            x = self.conv_block(x, self.bachnorm_momentum, conv=mid_layers[2])

            # decoding stage
            for l in range(self.num_layers):
                x = concatenate([x, down_layers[-1-l]])  # forward skip connection
                x = self.conv_block(x, self.bachnorm_momentum, conv = deconv_layers[3*l])
                x = self.conv_block(x, self.bachnorm_momentum, conv = deconv_layers[3*l+1])
                back_layers.append(x) 
                x = self.conv_block(x, self.bachnorm_momentum, conv = deconv_layers[3*l+2])

            #integrate decoded features
            if self.integrate:
                collection.append(x)
        
        #'integrate' leads to potential better results
        if self.integrate:
            x = concatenate(collection)
            
        #the last stage block is not reusable
        x = self.conv_block(x, self.bachnorm_momentum, int(32*self.multiplier))
        x = self.conv_block(x, self.bachnorm_momentum, int(32*self.multiplier))

        outputs = Conv2D(self.num_classes, kernel_size=(1,1), strides=(1,1), activation='sigmoid', padding='valid')(x)

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
        x = BatchNormalization(momentum=bachnorm_momentum)(x)  # BNs are NOT reusable
        return x          
