import torch
import torch.nn as nn
from torch.nn import init
import functools

class EncoderDecoderGenerator(nn.Module):
    """Create a EncoderDecoder-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in EncoderDecoder. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(EncoderDecoderGenerator, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        # downsampling / encoder
        self.net = [
            nn.Conv2d(input_nc, ngf, kernel_size=4,
                                stride=2, padding=1, bias=use_bias)
        ]
        for i in range(num_downs - 5):
            self.net.extend([
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ngf*int(2**i), ngf*int(2**(i+1)), kernel_size=4,
                                    stride=2, padding=1, bias=use_bias),
                norm_layer(ngf*int(2**(i+1)))
            ])
        i += 1
        self.net.extend([
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf*int(2**i), ngf*int(2**i), kernel_size=4,
                                stride=2, padding=1, bias=use_bias),
            norm_layer(ngf*int(2**i)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf*int(2**i), ngf*int(2**i), kernel_size=4,
                                stride=2, padding=1, bias=use_bias),
            norm_layer(ngf*int(2**i)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf*int(2**i), ngf*int(2**i), kernel_size=4,
                                stride=2, padding=1, bias=use_bias),
            norm_layer(ngf*int(2**i)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf*int(2**i), ngf*int(2**i), kernel_size=4,
                                stride=2, padding=1, bias=use_bias)
        ])
        
        # upsampling / decoder
        
        self.net.extend([
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*int(2**i), ngf*int(2**i),
                                    kernel_size=4, stride=2,
                                    padding=1, bias=use_bias),
            norm_layer(ngf*int(2**i)),
            nn.Dropout(0.5),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*int(2**i), ngf*int(2**i),
                                    kernel_size=4, stride=2,
                                    padding=1, bias=use_bias),
            norm_layer(ngf*int(2**i)),
            nn.Dropout(0.5),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*int(2**i), ngf*int(2**i),
                                    kernel_size=4, stride=2,
                                    padding=1, bias=use_bias),
            norm_layer(ngf*int(2**i)),
            nn.Dropout(0.5),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*int(2**i), ngf*int(2**i),
                                    kernel_size=4, stride=2,
                                    padding=1, bias=use_bias),
            norm_layer(ngf*int(2**i))
        ])

        for i in reversed(range(num_downs - 5)):
            self.net.extend([
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf*int(2**(i+1)), ngf*int(2**i),
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                norm_layer(ngf*int(2**i))
            ])
        self.net.extend([
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, output_nc,
                                    kernel_size=4, stride=2,
                                    padding=1, bias=use_bias),
            nn.Tanh()
        ])

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward"""
        return self.net(input)
