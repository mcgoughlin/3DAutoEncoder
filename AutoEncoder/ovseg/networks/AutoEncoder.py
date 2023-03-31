import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_padding(kernel_size):
    if isinstance(kernel_size, (list, tuple, np.ndarray)):
        # print([(k - 1) // 2 for k in kernel_size])
        return [(k - 1) // 2 for k in kernel_size]
    else:
        return (kernel_size - 1) // 2


def get_stride(kernel_size):
    if isinstance(kernel_size, (list, tuple, np.ndarray)):
        return [(k + 1)//2 for k in kernel_size]
    else:
        return (kernel_size + 1) // 2
    
def get_stride(kernel_size):
    if isinstance(kernel_size, (list, tuple, np.ndarray)):
        return [(k +1)//2 for k in kernel_size]
    else:
        return (kernel_size +1) // 2


# %%
class ConvNormNonlinBlock(nn.Module):

    def __init__(self, in_channels, out_channels, is_2d, kernel_size=3,
                 first_stride=1, conv_params=None, norm=None, norm_params=None,
                 nonlin_params=None, hid_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.hid_channels = hid_channels if hid_channels is not None else out_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = get_padding(self.kernel_size)
        self.first_stride = first_stride
        self.is_2d = is_2d
        self.conv_params = conv_params
        self.norm_params = norm_params
        self.nonlin_params = nonlin_params

        if norm is None:
            norm = 'batch' if is_2d else 'inst'

        if self.conv_params is None:
            self.conv_params = {'bias': False}
        if self.nonlin_params is None:
            self.nonlin_params = {'negative_slope': 0.01, 'inplace': True}
        if self.norm_params is None:
            self.norm_params = {'affine': True}
        # init convolutions, normalisation and nonlinearities
        if self.is_2d:
            conv_fctn = nn.Conv2d
            if norm.lower().startswith('batch'):
                norm_fctn = nn.BatchNorm2d
            elif norm.lower().startswith('inst'):
                norm_fctn = nn.InstanceNorm2d
        else:
            conv_fctn = nn.Conv3d
            if norm.lower().startswith('batch'):
                norm_fctn = nn.BatchNorm3d
            elif norm.lower().startswith('inst'):
                norm_fctn = nn.InstanceNorm3d
        # print("padding",self.padding)
        self.conv1 = conv_fctn(self.in_channels, self.hid_channels,
                               self.kernel_size, padding=self.padding,
                               stride=self.first_stride, **self.conv_params)
        self.conv2 = conv_fctn(self.hid_channels, self.out_channels,
                               self.kernel_size, padding=self.padding,
                               **self.conv_params)
        self.norm1 = norm_fctn(self.hid_channels, **self.norm_params)
        self.norm2 = norm_fctn(self.out_channels, **self.norm_params)
        

        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)

    def forward(self, xb):
        # print(xb.shape,"initial")
        xb = self.conv1(xb)
        # print(xb.shape,"first conv")
        xb = self.norm1(xb)
        xb = self.conv2(xb)
        # print(xb.shape,"second conv")
        xb = self.norm2(xb)
        # print("normConv",xb.shape)
        return xb


# %% transposed convolutions
class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels, is_2d, kernel_size=2):
        super().__init__()
        # print("kernel size:",kernel_size)
        if is_2d:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels,
                                           kernel_size, stride=kernel_size,
                                           bias=False)
        else:
            self.conv = nn.ConvTranspose3d(in_channels, out_channels,
                                           kernel_size, stride=kernel_size,
                                           bias=False)
            
        # print(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, xb):
        # print("UpConv before",xb.shape)
        # print("UpConv after",self.conv(xb).shape)
        return self.conv(xb)


class UpLinear(nn.Module):

    def __init__(self, kernel_size, is_2d):
        
        if is_2d:
            self.up = nn.Upsample(scale_factor=kernel_size, mode='bilinear',
                                  align_corners=True)
        else:
            self.up = nn.Upsample(scale_factor=kernel_size, mode='trilinear',
                                  align_corners=True)
    def forward(self, xb):
        return self.up(xb)

# %% now simply the logits
class Logits(nn.Module):

    def __init__(self, in_channels, out_channels, is_2d, p_dropout=0):
        super().__init__()
        if is_2d:
            self.logits = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            self.dropout = nn.Dropout2d(p_dropout, inplace=True)
        else:
            self.logits = nn.Conv3d(in_channels, out_channels, 1, bias=False)
            self.dropout = nn.Dropout3d(p_dropout, inplace=True)
        nn.init.kaiming_normal_(self.logits.weight)

    def forward(self, xb):
        return self.dropout(self.logits(xb))

# %%
class res_skip(nn.Module):

    def forward(self, xb1, xb2):
        return xb1 + xb2

class param_res_skip(nn.Module):

    def __init__(self, in_channels, is_2d):
        super().__init__()

        if is_2d:
            self.a = nn.Parameter(torch.zeros((1, in_channels, 1, 1)))
        else:
            self.a = nn.Parameter(torch.zeros((1, in_channels, 1, 1, 1)))

    def forward(self, xb_up, xb_skip):
        return xb_up + self.a * xb_skip

class scaled_res_skip(nn.Module):

    def __init__(self):
        super().__init__()

        self.a = nn.Parameter(torch.zeros(()))
    def forward(self, xb_up, xb_skip):
        return xb_up + self.a * xb_skip

# %%
class AutoEncoder(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_sizes,
                 is_2d, filters=32, filters_max=384, n_pyramid_scales=None,
                 conv_params=None, norm=None, norm_params=None, nonlin_params=None,
                 kernel_sizes_up=None, use_trilinear_upsampling=False,
                 use_less_hid_channels_in_decoder=False, fac_skip_channels=1,
                 p_dropout_logits=0.0, stem_kernel_size=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        # print("Kernel sizes:",self.kernel_sizes)
        self.is_2d = is_2d
        self.n_stages = len(kernel_sizes)
        self.filters = filters
        self.filters_max = filters_max
        self.conv_params = conv_params
        self.norm = norm
        self.norm_params = norm_params
        self.nonlin_params = nonlin_params
        self.kernel_sizes_up = kernel_sizes_up if kernel_sizes_up is not None else kernel_sizes[:-1]

        self.use_trilinear_upsampling = use_trilinear_upsampling
        self.use_less_hid_channels_in_decoder = use_less_hid_channels_in_decoder
        assert fac_skip_channels <= 1 and fac_skip_channels > 0
        self.fac_skip_channels = fac_skip_channels
        self.p_dropout_logits = p_dropout_logits
        self.stem_kernel_size = stem_kernel_size

        
        # we half the amount of channels every downsampling step
        self.filters_list = [max([self.filters//(4**i), 1])
                             for i in range(self.n_stages)]
        self.out_channels_down_list = self.filters_list
        self.skip_channels = [int(self.fac_skip_channels * ch)
                              for ch in self.out_channels_down_list[:-1]]
        # first let's make the lists for the blocks on both pathes
        # number of input and output channels of the blocks on the contracting path
        if self.stem_kernel_size is None:
            self.in_channels_down_list = [self.in_channels] + self.filters_list[:-1]
        else:
            self.in_channels_down_list = [self.filters] + self.filters_list[:-1]
        
        self.first_stride_list = [1] + [get_stride(ks) for ks in self.kernel_sizes[:-1]]

        # if we do trilinear upsampling we have to make sure that the right number of channels is
        # outputed from the block below
        if self.use_trilinear_upsampling:
            self.out_channels_up_list = [self.filters // 2]
        else:
            self.out_channels_up_list = self.filters_list

        self.hid_channels_up_list = self.filters_list[:-1]

        # determine how many scales on the upwars path with be connected to
        # a loss function
        if n_pyramid_scales is None:
            self.n_pyramid_scales = max([1, self.n_stages - 2])
        else:
            self.n_pyramid_scales = int(n_pyramid_scales)

        # now all the logits
        self.logits_in_list = self.out_channels_up_list[:self.n_pyramid_scales]
        self.in_channels_up_list = [2 * n_ch for n_ch in self.skip_channels]
        # blocks on the contracting path
        self.blocks_down = []
        for in_channels, out_channels, kernel_size, first_stride in zip(self.in_channels_down_list,
                                                                        self.out_channels_down_list,
                                                                        self.kernel_sizes,
                                                                        self.first_stride_list):
            block = ConvNormNonlinBlock(in_channels=in_channels,
                                        out_channels=out_channels,
                                        is_2d=self.is_2d,
                                        kernel_size=kernel_size,
                                        first_stride=first_stride,
                                        conv_params=self.conv_params,
                                        norm=self.norm,
                                        norm_params=self.norm_params,
                                        nonlin_params=self.nonlin_params)
            self.blocks_down.append(block)

        # blocks on the upsampling path
        self.blocks_up = []
        for in_channels, out_channels, hid_channels, kernel_size in zip(self.in_channels_up_list,
                                                                        self.out_channels_up_list,
                                                                        self.hid_channels_up_list,
                                                                        self.kernel_sizes_up):
            block = ConvNormNonlinBlock(in_channels=int(in_channels/2),
                                        out_channels=out_channels,
                                        is_2d=self.is_2d,
                                        kernel_size=kernel_size,
                                        conv_params=self.conv_params,
                                        norm=self.norm,
                                        norm_params=self.norm_params,
                                        nonlin_params=self.nonlin_params,
                                        hid_channels=hid_channels)
            self.blocks_up.append(block)

        # upsaplings
        self.upsamplings = []
        if self.use_trilinear_upsampling:
            mode = 'bilinear' if self.is_2d else 'trilinear'
            for kernel_size in self.kernel_sizes[:-1]:         
                scaled_factor = tuple([(k+1)//2 for k in kernel_size])
                self.upsamplings.append(nn.Upsample(scale_factor=scaled_factor,
                                                    mode=mode,
                                                    align_corners=True))
        else:
            for in_channels, out_channels, kernel_size in zip(self.out_channels_up_list[1:],
                                                              self.skip_channels,
                                                              self.kernel_sizes):

                self.upsamplings.append(UpConv(in_channels=in_channels,
                                           out_channels=out_channels,
                                           is_2d=self.is_2d,
                                           kernel_size=get_stride(kernel_size)))

        # logits
        self.all_logits = []
        for in_channels in self.logits_in_list:
            self.all_logits.append(Logits(in_channels=in_channels,
                                          out_channels=self.out_channels,
                                          is_2d=self.is_2d,
                                          p_dropout=self.p_dropout_logits))

        # now important let's turn everything into a module list
        self.blocks_down = nn.ModuleList(self.blocks_down)
        self.blocks_up = nn.ModuleList(self.blocks_up)
        self.upsamplings = nn.ModuleList(self.upsamplings)
        self.all_logits = nn.ModuleList(self.all_logits)
        
        if self.stem_kernel_size is not None:
            if self.is_2d:
                self.stem = nn.Conv2d(self.in_channels,
                                      self.filters,
                                      self.stem_kernel_size,
                                      self.stem_kernel_size)
            else:
                self.stem = nn.Conv3d(self.in_channels,
                                      self.filters,
                                      self.stem_kernel_size,
                                      self.stem_kernel_size)
            self.final_up_conv = UpConv(self.filters,
                                        self.out_channels,
                                        self.is_2d,
                                        self.stem_kernel_size)                

    def forward(self, xb):
        if self.stem_kernel_size is not None:
            xb = self.stem(xb)
        # keep all out tensors from the contracting path
        xb_list = []
        logs_list = []
        # contracting path
        for block, skip_ch in zip(self.blocks_down, self.skip_channels):
            xb = block(xb)
            xb_list.append(xb[:, :skip_ch])

        # bottom block
        xb = self.blocks_down[-1](xb)
        # print(xb.shape, " bottleneck block shape")
        # expanding path without logits
        for i in range(self.n_stages - 2, self.n_pyramid_scales-1, -1):
            xb = self.upsamplings[i](xb)
            xb = self.blocks_up[i](xb)

        # expanding path with logits
        for i in range(self.n_pyramid_scales - 1, -1, -1):
            xb = self.upsamplings[i](xb)
            del xb_list[i]
            xb = self.blocks_up[i](xb)
            logs = self.all_logits[i](xb)
            logs_list.append(logs)

        if self.stem_kernel_size is not None:
            logs_list.append(self.final_up_conv(xb))

        # as we iterate from bottom to top we have to flip the logits list
        return logs_list[::-1]

    def update_prg_trn(self, param_dict, h, indx=None):
        if 'p_dropout_logits' in param_dict:
            p = (1 - h) * param_dict['p_dropout_logits'][0] + h * param_dict['p_dropout_logits'][1]
            for l in self.all_logits:
                l.dropout.p = p

def get_2d_UNet(in_channels, out_channels, n_stages, filters=32):
    kernel_sizes = [3 for _ in range(n_stages)]
    return AutoEncoder(in_channels, out_channels, kernel_sizes, True)


def get_3d_UNet(in_channels, out_channels, n_stages, n_2d_blocks, filters=32,filters_max=256):
    kernel_sizes = [(3,3,3) if i < n_2d_blocks else 3
                    for i in range(n_stages)]
    return AutoEncoder(in_channels, out_channels, kernel_sizes, False)


# %%
if __name__ == '__main__':
    gpu = torch.device('cpu')
    net_3d = get_3d_UNet(1, 1, 5, 2, 8).to(gpu)
    xb_3d = torch.randn((1, 1, 128, 128, 128), device=gpu)
    print('3d')
    with torch.no_grad():
        yb_3d = net_3d(xb_3d)
    print('Output shapes:')
    for log in yb_3d:
        print(log.shape)
