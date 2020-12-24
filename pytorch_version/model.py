from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, ConvTranspose2d, ReLU, MaxPool2d, Sigmoid, Parameter
from torch import tensor, cat


class BiONet(Module):

    def __init__(self,
                 num_classes: int = 1,
                 iterations: int = 2,
                 multiplier: float = 1.0,
                 num_layers: int = 4,
                 integrate: bool = False):

        super(BiONet, self).__init__()
        #  参数
        self.iterations = iterations
        self.multiplier = multiplier
        self.num_layers = num_layers
        self.integrate = integrate
        self.batch_norm_momentum = 0.01
        #  生成通道参数，此处通道是从第一个Encoder输出的量开始的，直到语义向量
        self.filters_list = [int(32 * (2 ** i) * self.multiplier) for i in range(self.num_layers + 1)]
        #  预处理卷积块，不参与循环，最终输出的是32*256*256
        self.pre_transform_conv_block = Sequential(
            # 这里看代码实现，应该永远和第一个Encoder输出的层数是一样的
            Conv2d(3, self.filters_list[0], kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),  # 生成f[1]*512*512
            ReLU(),
            BatchNorm2d(self.filters_list[0], momentum=self.batch_norm_momentum),
            Conv2d(self.filters_list[0], self.filters_list[0], kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            # 生成f[1]*512*512
            ReLU(),
            BatchNorm2d(self.filters_list[0], momentum=self.batch_norm_momentum),
            Conv2d(self.filters_list[0], self.filters_list[0], kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            # 生成f[1]*512*512
            ReLU(),
            BatchNorm2d(self.filters_list[0], momentum=self.batch_norm_momentum),
            MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        )
        self.reuse_convs = []  # encoder复用的卷积核，每个encoder对应一个元组（共3个卷积核，不包括ReLU）
        self.encoders = []  # 由encoder构成的列表。由于encoder的一部分不参与循环，因此每个encoder是一个元组(两个CONV的Sequential, DOWN)
        self.reuse_deconvs = []  # decoder复用的卷积、反卷积核，每个decoder对应一个元组（共3个卷积核，不包括ReLU）
        self.decoders = []  # 由decoder构成的列表。由于decoder的一部分不参与循环，因此每个decoder是一个元组(两个CONV的Sequential, UP)
        for iteration in range(self.iterations):
            for layer in range(self.num_layers):

                #  创建encoders的部分。虽然部分代码可以合写，但是为了看起来清晰（而且构造函数也不是很要求效率），所以分开考虑encoder和decoder
                #  和层次有关的常数
                in_channel = self.filters_list[layer] * 2  # 由于有输出部分传入的数据，因此需要翻倍输入通道
                mid_channel = self.filters_list[layer]
                out_channel = self.filters_list[layer + 1]
                #  创建encoders模型
                if iteration == 0:
                    #  创建并添加复用卷积核
                    #  只有最后一个卷积核负责升高通道
                    conv1 = Conv2d(in_channel, mid_channel, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
                    conv2 = Conv2d(mid_channel, mid_channel, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
                    conv3 = Conv2d(mid_channel, out_channel, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
                    self.reuse_convs.append((conv1, conv2, conv3))
                #  创建encoder
                #  首先构造两个CONV
                convs = Sequential(
                    self.reuse_convs[layer][0],
                    ReLU(),
                    BatchNorm2d(mid_channel, momentum=self.batch_norm_momentum),
                    self.reuse_convs[layer][1],
                    ReLU(),
                    BatchNorm2d(mid_channel, momentum=self.batch_norm_momentum)
                )
                #  构建DOWN
                down = Sequential(
                    self.reuse_convs[layer][2],
                    ReLU(),
                    BatchNorm2d(out_channel, momentum=self.batch_norm_momentum),
                    MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
                )
                self.add_module("iteration{0}_layer{1}_encoder_convs".format(iteration, layer), convs)
                self.add_module("iteration{0}_layer{1}_encoder_down".format(iteration, layer), down)
                self.encoders.append((convs, down))

                #  创建decoders的部分，仿照encoders
                #  和层次有关的常数，注意本部分不需要mid_channel，因为第一个卷积核就已经升高维度了
                in_channel = self.filters_list[self.num_layers - layer] + self.filters_list[self.num_layers - 1 - layer]
                out_channel = self.filters_list[self.num_layers - 1 - layer]
                #  创建decoders模型
                if iteration == 0:
                    #  创建并添加复用卷积核
                    #  从第一个卷积核就升高通道数
                    conv1 = Conv2d(in_channel, out_channel, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
                    conv2 = Conv2d(out_channel, out_channel, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
                    conv3 = ConvTranspose2d(out_channel, out_channel, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2),
                                            output_padding=(1, 1))  # 此处和tensorflow有点区别，为了完整的形状，需要用output补一补
                    self.reuse_deconvs.append((conv1, conv2, conv3))
                #  创建encoder
                #  首先构造两个CONV
                convs = Sequential(
                    self.reuse_deconvs[layer][0],
                    ReLU(),
                    BatchNorm2d(out_channel, momentum=self.batch_norm_momentum),
                    self.reuse_deconvs[layer][1],
                    ReLU(),
                    BatchNorm2d(out_channel, momentum=self.batch_norm_momentum)
                )
                #  构造UP
                up = Sequential(
                    self.reuse_deconvs[layer][2],
                    ReLU(),
                    BatchNorm2d(out_channel, momentum=self.batch_norm_momentum)
                )
                self.add_module("iteration{0}_layer{1}_decoder_convs".format(iteration, layer), convs)
                self.add_module("iteration{0}_layer{1}_decoder_up".format(iteration, layer), up)
                self.decoders.append((convs, up))
        #  创建middle层
        self.middles = Sequential(
            Conv2d(self.filters_list[-1], self.filters_list[-1], kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            ReLU(),
            BatchNorm2d(self.filters_list[-1], momentum=self.batch_norm_momentum),
            Conv2d(self.filters_list[-1], self.filters_list[-1], kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            ReLU(),
            BatchNorm2d(self.filters_list[-1], momentum=self.batch_norm_momentum),
            ConvTranspose2d(self.filters_list[-1], self.filters_list[-1], kernel_size=(3, 3), padding=(1, 1),
                            stride=(2, 2), output_padding=(1, 1)),
            ReLU(),
            BatchNorm2d(self.filters_list[-1], momentum=self.batch_norm_momentum)
        )
        self.post_transform_conv_block = Sequential(
            Conv2d(self.filters_list[0] * self.iterations, self.filters_list[0], kernel_size=(3, 3), padding=(1, 1),
                   stride=(1, 1)) if self.integrate else Conv2d(self.filters_list[0],
                                                                self.filters_list[0], kernel_size=(3, 3),
                                                                padding=(1, 1), stride=(1, 1)),
            ReLU(),
            BatchNorm2d(self.filters_list[0], momentum=self.batch_norm_momentum),
            Conv2d(self.filters_list[0], self.filters_list[0], kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            ReLU(),
            BatchNorm2d(self.filters_list[0], momentum=self.batch_norm_momentum),
            Conv2d(self.filters_list[0], 1, kernel_size=(1, 1), stride=(1, 1)),
            Sigmoid(),
        )

    def forward(self, x: tensor) -> tensor:
        enc = [None for i in range(self.num_layers)]
        dec = [None for i in range(self.num_layers)]
        all_output = [None for i in range(self.iterations)]
        x = self.pre_transform_conv_block(x)
        e_i = 0
        d_i = 0
        for iteration in range(self.iterations):
            for layer in range(self.num_layers):
                if layer == 0:
                    x_in = x
                x_in = self.encoders[e_i][0](cat([x_in, x_in if dec[-1 - layer] is None else dec[-1 - layer]], dim=1))
                enc[layer] = x_in
                x_in = self.encoders[e_i][1](x_in)
                e_i = e_i + 1
            x_in = self.middles(x_in)
            for layer in range(self.num_layers):
                x_in = self.decoders[d_i][0](cat([x_in, enc[-1 - layer]], dim=1))
                dec[layer] = x_in
                x_in = self.decoders[d_i][1](x_in)
                d_i = d_i + 1
            all_output[iteration] = x_in
        if self.integrate:
            x_in = cat(all_output, dim=1)
        x_in = self.post_transform_conv_block(x_in)
        return x_in