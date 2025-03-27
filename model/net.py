import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
import fjn_util
import os
# from mmcv.cnn import ConvModule
import warnings
warnings.filterwarnings('ignore')

from model import net_common as common
from mambaIR import MambaIR as MambaNet


# MDCB
class MDCB(nn.Module):
    def __init__(self, ch_in, ch_out, bias=True, activation=nn.ReLU(inplace=True)):
        super(MDCB, self).__init__()

        kernel_size_1 = 3
        kernel_size_2 = 5

        self.conv_3_1 = common.default_conv(ch_in=ch_in, ch_out=ch_in, kernel_size=kernel_size_1,  bias=bias)
        self.conv_3_2 = common.default_conv(ch_in=ch_out, ch_out=ch_out, kernel_size=kernel_size_1,  bias=bias)
        self.conv_5_1 = common.default_conv(ch_in=ch_in, ch_out=ch_in, kernel_size=kernel_size_2,  bias=bias)
        self.conv_5_2 = common.default_conv(ch_in=ch_out, ch_out=ch_out, kernel_size=kernel_size_2,  bias=bias)
        self.confusion_3 = nn.Conv2d(ch_in * 3, ch_out, 1, padding=0, bias=True)
        self.confusion_5 = nn.Conv2d(ch_in * 3, ch_out, 1, padding=0, bias=True)
        self.confusion_bottle = nn.Conv2d(ch_in * 3 + ch_out * 2, ch_out, 1, padding=0, bias=True)
        # self.relu = nn.ReLU(inplace=True)
        self.activation = activation

    def forward(self, x):
        input_1 = x  # [1, 3, 256, 256]
        output_3_1 = self.activation(self.conv_3_1(input_1))  # [1, 3, 256, 256]
        output_3_1 += x
        output_5_1 = self.activation(self.conv_5_1(input_1))  # [1, 3, 256, 256]
        output_5_1 += x
        input_2 = torch.cat([input_1, output_3_1, output_5_1], 1)  # [1, 9, 256, 256]

        input_2_3 = self.confusion_3(input_2)
        input_2_5 = self.confusion_5(input_2)

        output_3_2 = self.activation(self.conv_3_2(input_2_3))
        output_5_2 = self.activation(self.conv_5_2(input_2_5))
        input_3 = torch.cat([input_1, output_3_1, output_5_1, output_3_2, output_5_2], 1)
        output = self.confusion_bottle(input_3)
        # output += x
        return output  # [1, 3, 256, 256]
    

class IMUB_Head(nn.Module):
    def __init__(self, ch_in, ch_out, activation=nn.ReLU(), bias=False, down_times=2, fe_num=16):
        super(IMUB_Head, self).__init__()

        self.down_times = down_times

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=ch_in, out_channels=fe_num, kernel_size=1, stride=1,padding=0, bias=bias),
            activation,
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=fe_num, out_channels=fe_num, kernel_size=3, stride=1,padding=1, bias=bias),
            activation,
        )

        self.conv2_1 = nn.Sequential(
            nn.Conv2d(in_channels=ch_in, out_channels=fe_num, kernel_size=1, stride=1,padding=0, bias=bias),
            activation,
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(in_channels=fe_num, out_channels=fe_num, kernel_size=3, stride=1,padding=1, bias=bias),
            activation,
        )
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(in_channels=fe_num, out_channels=fe_num, kernel_size=3, stride=1,padding=1, bias=bias),
            activation,
        )

        self.conv3_1 = nn.Sequential(
            nn.Conv2d(in_channels=ch_in, out_channels=fe_num, kernel_size=1, stride=1,padding=0, bias=bias),
            activation,
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(in_channels=fe_num, out_channels=fe_num, kernel_size=3, stride=1,padding=1, bias=bias),
            activation,
        )
        self.conv3_3 = nn.Sequential(
            nn.Conv2d(in_channels=fe_num, out_channels=fe_num, kernel_size=3, stride=1,padding=1, bias=bias),
            activation,
        )
        self.conv3_4 = nn.Sequential(
            nn.Conv2d(in_channels=fe_num, out_channels=fe_num, kernel_size=3, stride=1,padding=1, bias=bias),
            activation,
        )

        self.conv_end = nn.Sequential(
            nn.Conv2d(in_channels=fe_num * 3, out_channels=ch_out, kernel_size=1, stride=1, padding=0, bias=bias),
            activation,
        )

        self.conv_sum = nn.Sequential(
            nn.Conv2d(in_channels=ch_out * (down_times + 1), out_channels=ch_out, kernel_size=1, stride=1, padding=0, bias=bias),
            activation,
        )

        self.activation = activation

    def forward(self, x):
        b, _, h, w = x.size()
        input, features, result = [], [], []
        input.append(x)

        for i in range(self.down_times):
            input.append(F.upsample(x, size=(x.shape[2] // (2*i + 2), x.shape[3] // (2*i + 2)), mode='bilinear'))

        for item in input:

            x1 = self.conv1_1(item)
            x1 = self.conv1_2(x1) + x1

            x2 = self.conv2_1(item)
            x2 = self.conv2_2(x2) + x2
            x2 = self.conv2_3(x2) + x2

            x3 = self.conv3_1(item)
            x3 = self.conv3_2(x3) + x3
            x3 = self.conv3_3(x3) + x3
            x3 = self.conv3_4(x3) + x3

            out = torch.cat([x1, x2,x3], dim=1)
            out = self.conv_end(out)
            features.append(out)

        for item in features:
            if item.shape[2] != h or item.shape[3] != w:
                result.append(F.upsample(item, size=(h, w), mode='bilinear'))
            else:
                result.append(item)

        out = result[0]
        for i in range(1, len(result)):
            out = torch.cat([out, result[i]], dim=1)

        del input, features, result
        out = self.conv_sum(out)

        return out

# IDB
class IDB(nn.Module):

    def __init__(self, in_planes, planes, bias, activation):
        super(IDB, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=3, padding=1, stride=1, bias=bias),
            activation,
            common.involution(channels=planes, kernel_size=3, stride=2),
            activation,
            nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=1, padding=0, stride=1, bias=bias),
            activation,
        )

    def forward(self, x):
        out = self.cnn(x)
        return out


# IMUB
class IMUB(nn.Module):

    def __init__(self, in_planes, planes, bias, activation):
        super(IMUB, self).__init__()

        self.cnn = nn.Sequential(
            IMUB_Head(ch_in=in_planes, ch_out=planes, bias=bias, activation=activation),
            common.ConvUpsampler(ch_in=planes, ch_out=planes, activation=activation, bias=bias),
            activation,
        )

    def forward(self, x):
        x = self.cnn(x)
        # x = F.upsample(x, size=(x.shape[2] * 2, x.shape[3] * 2), mode='bilinear')
        return x

class Multi_SEAttention(nn.Module):  # 定义SE注意力
    def __init__(self, in_planes, reduction=16):
        super(Multi_SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(in_planes, in_planes // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // reduction, in_planes, bias=False),
            nn.Sigmoid()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_planes, in_planes // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // reduction, in_planes, bias=False),
            nn.Sigmoid()
        )

        # self.sum = nn.Conv2d(in_planes * 3, in_planes, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y).view(b, c, 1, 1)

        z = self.max_pool(x).view(b, c)
        z = self.fc2(z).view(b, c, 1, 1)
        x1 = x * y.expand_as(x)
        x2 = x * z.expand_as(x)
        x_sum = x1 + x2 + x
        return x_sum


class No_Multi_SEAttention(nn.Module):  # 定义SE注意力
    def __init__(self, in_planes, reduction=16):
        super(No_Multi_SEAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        return out

class UNet_Plus_Residual(nn.Module):
    def __init__(self, down_block, se_block, up_block, end_block, grow_rate=32, depth=4, fe_num=16, bias=False, multi_out=False, scale=1):
        super(UNet_Plus_Residual, self).__init__()

        self.depth = depth
        self.fe_num = fe_num
        self.grow_rate = grow_rate
        self.multi_out = multi_out
        self.scale = scale

        self.bias = bias
        self.activation = common.Mish()

        self.root = self.make_head()

        self.layers = self._make_layers(depth=self.depth, in_planes=fe_num, down_block=down_block, up_block=up_block, se_block=se_block)

        if self.multi_out:
            self.end_convs = self._make_end_convs(self.depth, end_block)
        else:
            self.end_convs = self._make_end_convs(1, end_block)
        self.end_convs = self._make_end_convs(self.depth, end_block)

    def make_head(self):
        if self.scale == 1:
            head = nn.Sequential(
                MDCB(3, 32, bias=self.bias, activation=self.activation),
                MDCB(32, self.fe_num, bias=self.bias, activation=self.activation),
            )
        elif self.scale == 2:
            head = nn.Sequential(
                MDCB(3, 32, bias=self.bias, activation=self.activation),
                MDCB(32, self.fe_num, bias=self.bias, activation=self.activation),
                # common.ConvUpsampler(self.fe_num, self.fe_num),
                MambaNet(img_size=64, patch_size=1, in_chans=self.fe_num, embed_dim=180, depths=(6, 6, 6, 6, 6, 6),
                                      mlp_ratio=2.,
                                      drop_rate=0., norm_layer=nn.LayerNorm, patch_norm=True, use_checkpoint=False,
                                      upscale=2, img_range=1.,
                                      upsampler='pixelshuffle', resi_connection='1conv')
            )
        elif self.scale == 4:
            head = nn.Sequential(
                MDCB(3, 32, bias=self.bias, activation=self.activation),
                MDCB(32, self.fe_num, bias=self.bias, activation=self.activation),
                common.ConvUpsampler(self.fe_num, self.fe_num),
                common.ConvUpsampler(self.fe_num, self.fe_num),
            )
        elif self.scale == 8:
            head = nn.Sequential(
                MDCB(3, 32, bias=self.bias, activation=self.activation),
                MDCB(32, self.fe_num, bias=self.bias, activation=self.activation),
                common.ConvUpsampler(self.fe_num, self.fe_num),
                common.ConvUpsampler(self.fe_num, self.fe_num),
                common.ConvUpsampler(self.fe_num, self.fe_num),
            )
        return head

    def _make_layers(self, depth, in_planes, down_block, up_block, se_block):

        layers = []
        in_planes = in_planes
        planes = in_planes + self.grow_rate
        for i in range(depth):
            layer_list = []
            if i == 0:
                layers_planes = planes
                layer_list.append(down_block(layers_planes - self.grow_rate, layers_planes, self.bias, self.activation))
                layer_list.append(up_block(layers_planes, layers_planes - self.grow_rate, self.bias, self.activation))
                if se_block is not None:
                    layer_list.append(se_block(layers_planes - self.grow_rate))
                layers.append(nn.Sequential(*layer_list))
                planes += self.grow_rate
            else:
                layers_planes = planes
                for j in range(i + 1):
                    if j == 0:
                        layer_list.append( down_block(layers_planes - self.grow_rate, layers_planes, self.bias, self.activation))
                    layer_list.append( up_block(layers_planes, layers_planes - self.grow_rate, self.bias, self.activation))
                    if se_block is not None:
                        layer_list.append(se_block(layers_planes - self.grow_rate))
                    layers_planes -= self.grow_rate
                layers.append(nn.Sequential(*layer_list))
                planes += self.grow_rate
        return nn.Sequential(*layers)

    def _make_end_convs(self, depth, block):
        conv_list = []
        for i in range(depth):
            conv_list.append(block(ch_in=self.fe_num, ch_out=3))
        return nn.Sequential(*conv_list)

    def forward(self, input):

        # features
        features = []
        x = self.root(input)
        features.append([x])

        for i in range(self.depth):
            # layer 1
            if i == 0:
                y_11 = self.layers[0][0](x)
                y_12 = self.layers[0][1](y_11) + self.layers[0][2](x) + x
                features.append([y_11, y_12])
            # layer 2 -
            else:
                features_layer = []
                layer_step = 0
                feature_step = 0
                for j in range(i + 1):
                    # print('i = {} , j = {} '.format(i,j))
                    if j == 0:
                        y_n1 = self.layers[i][0](features[i][0])
                        y_n2 = self.layers[i][1](y_n1) + self.layers[i][2](features[i][0])+features[i][0]
                        features_layer.append(y_n1)
                        features_layer.append(y_n2)
                        layer_step = 3
                        feature_step = 1
                    else:
                        y_nm = self.layers[i][layer_step](features_layer[-1]) + self.layers[i][layer_step + 1](features[i][feature_step])+features[i][feature_step]
                        features_layer.append(y_nm)
                        layer_step += 2
                        feature_step += 1
                features.append(features_layer)

        # result
        result = []
        for item in features:
            result.append(item[-1])

        # out
        out = []
        if self.multi_out:
            for i, item in enumerate(result):
                if i == 0:
                    continue
                y = self.activation(self.end_convs[i - 1](result[i]))
                out.append(y)
        else:
            y = self.activation(self.end_convs[0](result[-1]))
            out.append(y)


        return out

    def model_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    m.bias.data.fill_(0)
        pass

    def load_pkl(self,pkl_path,mode='trained',load = True):
        '''

        :param pkl_path: pkl路径
        :param mode: trained or best
        :param load: bool,是否要加载模型（训练时不用加载最佳模型）
        :return:
        '''
        success_load = False
        if not os.path.exists(pkl_path):
            print('PKL:this pkl path dose not exit!')
        else:
            # 寻找pkl文件并加载
            pkl_list = os.listdir(pkl_path)  # 列出路径下所有的文件名称

            if len(pkl_list) == 0:
                print('PKL:no ' + mode + ' model!')
            elif pkl_list[-1].find('.pkl') == -1:
                print('PKL:no ' + mode + ' model!')
            else:
                checkpoint = torch.load(pkl_path + pkl_list[-1])
                if load:
                    self.load_state_dict(checkpoint['model_state_dict'])
                success_load = True
        if success_load:
            if mode == 'trained':
                self.train_epoch = checkpoint['epoch']
                print(mode.capitalize() + ' PKL(EPOCH:'+str(self.train_epoch - 1)+') successfully loaded !')
            elif mode == 'best':
                self.best_epoch = checkpoint['epoch']
                self.best_psnr = checkpoint['psnr']
                self.best_ssim = checkpoint['ssim']
                print(mode.capitalize() + ' PKL(EPOCH:' + str(self.best_epoch - 1) + ') successfully loaded !')
        else:
            if mode == 'trained':
                self.train_epoch = 1
            elif mode == 'best':
                self.best_epoch = 1
                self.best_psnr = 0
                self.best_ssim = 0
        return self

def Kong(depth=4, grow_rate=32, fe_num=128, multi_out=False, scale=2):
    net = UNet_Plus_Residual(
        down_block=IDB,
        se_block=No_Multi_SEAttention,
        up_block=IMUB,
        end_block=common.Default_Conv,
        depth=depth, grow_rate=grow_rate, fe_num=fe_num, bias=True, multi_out=multi_out, scale=scale)
    return net

def find_last_pkl(pkl_path):
    pkl_src = None

    # 1. not find pklpath
    if not os.path.exists(pkl_path):
        fjn_util.make_folder(pkl_path)
        print('PKL: this path does not exist!')
    else:
        pkl_list = os.listdir(pkl_path)

        # 2. no files in pklpath
        if len(pkl_list) == 0:
            print('PKL: ', pkl_path, 'no trained moudle exist(no any files)!')
        else:
            index = pkl_list[-1].find('.pkl')

            # 3. not find pkl in path
            if not index:
                print('TRAIN PKL: ', pkl_path, 'no trained moudle exist(no pkl files)!')
            else:
                # 4. load pkl
                pkl_src = pkl_path + pkl_list[-1]

    return pkl_src

def load_train_model(self):
    pkl_src = self.find_last_pkl(self.train_pkl_path)
    if pkl_src:
        checkpoint = torch.load(pkl_src)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.train_epoch = checkpoint['epoch']
        print('TRAIN PKL: {} load successfully!'.format(pkl_src))
    else:
        print('not load train model!')
        if self.init:
            self.model_init()
            print('model! initliaze...')
        self.train_epoch = 0
    self.train_ssim = 0

def load_best_model(model, best_pkl_path):
    pkl_src = find_last_pkl(best_pkl_path)
    if pkl_src:
        checkpoint = torch.load(pkl_src)
        best_epoch = checkpoint['best_epoch']
        best_psnr = checkpoint['best_psnr']
        best_ssim = checkpoint['best_ssim']
        model.load_state_dict(checkpoint['model_state_dict'])
        print('BEST PKL: {} load successfully!'.format(pkl_src))
    else:
        best_epoch = 1
        best_psnr = 0
        best_ssim = 0
        print('not load best model!')
    return model, best_epoch, best_psnr, best_ssim

# def test():
#     model = Kong(scale=1).cuda()
#     input = torch.rand((1,3,64,64)).cuda()
# 
#     checkpoint = torch.load('../pretrain/color.pkl')
#     model.load_state_dict(checkpoint['model_state_dict'])
#     result = model(input)
#     print('shape: ', result[0].shape)
# 
# 
# 
# test()
