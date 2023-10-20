import torch
import torch.nn as nn
import torch.nn.functional as F


class UpSampleConv2D(torch.jit.ScriptModule):
    def __init__(
        self,
        input_channels,
        kernel_size=3,
        n_filters=128,
        upscale_factor=2,
        padding=0,
    ):
        super(UpSampleConv2D, self).__init__()
        self.conv = nn.Conv2d(
            input_channels, n_filters, kernel_size=kernel_size, padding=padding
        )
        self.upscale_factor = upscale_factor
        self.pixel_shuffle = nn.PixelShuffle(int(self.upscale_factor))
        self.pixel_shuffle = nn.PixelShuffle(int(self.upscale_factor))

    @torch.jit.script_method
    def forward(self, x):
        ##################################################################
        # TODO 1.1: Implement nearest neighbor upsampling
        # 1. Repeat x channel-wise upscale_factor^2 times
        # 2. Use torch.nn.PixelShuffle to form an output of dimension
        # (batch, channel, height*upscale_factor, width*upscale_factor)
        # 3. Apply convolution and return output
        ##################################################################
        x_repeated = x.repeat(1,int(self.upscale_factor**2),1,1)
        x_scaled = self.pixel_shuffle(x_repeated)
        return self.conv(x_scaled)
        x_repeated = x.repeat(1,int(self.upscale_factor**2),1,1)
        x_scaled = self.pixel_shuffle(x_repeated)
        return self.conv(x_scaled)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################


class DownSampleConv2D(torch.jit.ScriptModule):
    def __init__(
        self, input_channels, kernel_size=3, n_filters=128, downscale_ratio=2, padding=0
    ):
        super(DownSampleConv2D, self).__init__()
        self.conv = nn.Conv2d(
            input_channels, n_filters, kernel_size=kernel_size, padding=padding
        )
        self.downscale_ratio = downscale_ratio
        self.pixel_unshuffle = nn.PixelUnshuffle(int(self.downscale_ratio))
        self.pixel_unshuffle = nn.PixelUnshuffle(int(self.downscale_ratio))

    @torch.jit.script_method
    def forward(self, x):
        ##################################################################
        # TODO 1.1: Implement spatial mean pooling
        # 1. Use torch.nn.PixelUnshuffle to form an output of dimension
        # (batch, channel*downscale_factor^2, height, width)
        # 2. Then split channel-wise and reshape into
        # (downscale_factor^2, batch, channel, height, width) images
        # 3. Take the average across dimension 0, apply convolution,
        # and return the output
        ##################################################################
        x = self.pixel_unshuffle(x)
        channel = x.shape[1]
        x = torch.stack(x.split(channel//int(self.downscale_ratio**2),dim=1))
        x = torch.mean(x, dim=0)
        return self.conv(x)
        x = self.pixel_unshuffle(x)
        channel = x.shape[1]
        x = torch.stack(x.split(channel//int(self.downscale_ratio**2),dim=1))
        x = torch.mean(x, dim=0)
        return self.conv(x)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################


class ResBlockUp(torch.jit.ScriptModule):
    """
    ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(n_filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
                (conv): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(input_channels, n_filters, kernel_size=(1, 1), stride=(1, 1))
        )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlockUp, self).__init__()
        ##################################################################
        # TODO 1.1: Setup the network layers
        ##################################################################
        self.layers = nn.Sequential(nn.BatchNorm2d(num_features=input_channels,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True),
                       nn.ReLU(),
                       nn.Conv2d(in_channels=input_channels,out_channels=n_filters,kernel_size=3,stride=1,padding=1,bias=False),
                       nn.BatchNorm2d(num_features=n_filters,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True),
                       nn.ReLU(),
                       UpSampleConv2D(input_channels=n_filters,n_filters=n_filters,kernel_size=1))
        self.upsample_residual = UpSampleConv2D(input_channels=input_channels,n_filters=n_filters,kernel_size=1)
        self.layers = nn.Sequential(nn.BatchNorm2d(num_features=input_channels,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True),
                       nn.ReLU(),
                       nn.Conv2d(in_channels=input_channels,out_channels=n_filters,kernel_size=3,stride=1,padding=1,bias=False),
                       nn.BatchNorm2d(num_features=n_filters,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True),
                       nn.ReLU(),
                       UpSampleConv2D(input_channels=n_filters,n_filters=n_filters,kernel_size=1))
        self.upsample_residual = UpSampleConv2D(input_channels=input_channels,n_filters=n_filters,kernel_size=1)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    @torch.jit.script_method
    def forward(self, x):
        ##################################################################
        # TODO 1.1: Forward through the layers and implement a residual
        # connection. Make sure to upsample the residual before adding it
        # to the layer output.
        ##################################################################
        input = x
        x = self.layers(x)
        return self.upsample_residual(input)+x
        input = x
        x = self.layers(x)
        return self.upsample_residual(input)+x
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################


class ResBlockDown(torch.jit.ScriptModule):
    """
    ResBlockDown(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): DownSampleConv2D(
                (conv): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (downsample_residual): DownSampleConv2D(
            (conv): Conv2d(input_channels, n_filters, kernel_size=(1, 1), stride=(1, 1))
        )
    )
    """
    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlockDown, self).__init__()
        ##################################################################
        # TODO 1.1: Setup the network layers
        ##################################################################
        self.layers = nn.Sequential(nn.ReLU(),
                                    nn.Conv2d(in_channels=input_channels,out_channels=n_filters,kernel_size=3,stride=1,padding=1),
                                    nn.ReLU(),
                                    DownSampleConv2D(input_channels=n_filters,kernel_size=3,n_filters=n_filters,padding=1))
        self.downsample_residual = DownSampleConv2D(input_channels=input_channels,kernel_size=1,n_filters=n_filters)
        self.layers = nn.Sequential(nn.ReLU(),
                                    nn.Conv2d(in_channels=input_channels,out_channels=n_filters,kernel_size=3,stride=1,padding=1),
                                    nn.ReLU(),
                                    DownSampleConv2D(input_channels=n_filters,kernel_size=3,n_filters=n_filters,padding=1))
        self.downsample_residual = DownSampleConv2D(input_channels=input_channels,kernel_size=1,n_filters=n_filters)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    @torch.jit.script_method
    def forward(self, x):
        ##################################################################
        # TODO 1.1: Forward through the layers and implement a residual
        # connection. Make sure to downsample the residual before adding
        # it to the layer output.
        ##################################################################
        input = x
        x = self.layers(x)
        down_input = self.downsample_residual(input)
        return down_input+x
        input = x
        x = self.layers(x)
        down_input = self.downsample_residual(input)
        return down_input+x
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################


class ResBlock(torch.jit.ScriptModule):
    """
    ResBlock(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
    )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlock, self).__init__()
        ##################################################################
        # TODO 1.1: Setup the network layers
        ##################################################################
        self.layers = nn.Sequential(nn.ReLU(),
                                    nn.Conv2d(in_channels=input_channels,out_channels=n_filters,kernel_size=3,stride=1,padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=n_filters,out_channels=n_filters,kernel_size=3,stride=1,padding=1))
        self.layers = nn.Sequential(nn.ReLU(),
                                    nn.Conv2d(in_channels=input_channels,out_channels=n_filters,kernel_size=3,stride=1,padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=n_filters,out_channels=n_filters,kernel_size=3,stride=1,padding=1))
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    @torch.jit.script_method
    def forward(self, x):
        ##################################################################
        # TODO 1.1: Forward the conv layers. Don't forget the residual
        # connection!
        ##################################################################
        input = x
        return input + self.layers(x)
        input = x
        return input + self.layers(x)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

class Generator(torch.jit.ScriptModule):
    """
    Generator(
    (dense): Linear(in_features=128, out_features=2048, bias=True)
    (layers): Sequential(
        (0): ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (1): ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (2): ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (4): ReLU()
        (5): Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (6): Tanh()
    )
    )
    """

    def __init__(self, starting_image_size=4):
        super(Generator, self).__init__()
        ##################################################################
        # TODO 1.1: Set up the network layers. You should use the modules
        # you have implemented previously above.
        ##################################################################
        self.dense = nn.Linear(128, out_features=2048, bias=True)
        self.layers = nn.Sequential(ResBlockUp(input_channels=128),ResBlockUp(128),ResBlockUp(128))
        self.bn = nn.BatchNorm2d(num_features=128,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels=128,out_channels=3,kernel_size=3,stride=1,padding=1)
        self.activation = nn.Tanh()
        self.starting_image_size = starting_image_size
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    @torch.jit.script_method
    def forward_given_samples(self, z):
        ##################################################################
        # TODO 1.1: Forward the generator assuming a set of samples z has
        # been passed in. Don't forget to re-shape the output of the dense
        # layer into an image with the appropriate size!
        ##################################################################
        batch_size = z.shape[0]
        z = z.reshape(batch_size,-1)
        z = self.dense(z).reshape(batch_size,128,4,4)
        z = self.layers(z)
        z = self.bn(z)
        z = self.relu(z)
        z = self.conv(z)
        z = self.activation(z)
        return z
        batch_size = z.shape[0]
        z = z.reshape(batch_size,-1)
        z = self.dense(z).reshape(batch_size,128,4,4)
        z = self.layers(z)
        z = self.bn(z)
        z = self.relu(z)
        z = self.conv(z)
        z = self.activation(z)
        return z
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    @torch.jit.script_method
    def forward(self, n_samples: int = 1024):
        ##################################################################
        # TODO 1.1: Generate n_samples latents and forward through the
        # network.
        ##################################################################
        z = torch.randn((n_samples,128)).cuda()
        z = self.forward_given_samples(z)
        return z
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################


class Discriminator(torch.jit.ScriptModule):
    """
    Discriminator(
    (layers): Sequential(
        (0): ResBlockDown(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): DownSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (downsample_residual): DownSampleConv2D(
            (conv): Conv2d(3, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (1): ResBlockDown(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): DownSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (downsample_residual): DownSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (2): ResBlock(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        )
        (3): ResBlock(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        )
        (4): ReLU()
    )
    (dense): Linear(in_features=128, out_features=1, bias=True)
    )
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        ##################################################################
        # TODO 1.1: Set up the network layers. You should use the modules
        # you have implemented previously above.
        ##################################################################
        self.dense = nn.Linear(in_features=128,out_features=1,bias=True)
        self.layers = nn.Sequential(ResBlockDown(input_channels=3),ResBlockDown(128),ResBlock(128),ResBlock(128),nn.ReLU())
        self.dense = nn.Linear(in_features=128,out_features=1,bias=True)
        self.layers = nn.Sequential(ResBlockDown(input_channels=3),ResBlockDown(128),ResBlock(128),ResBlock(128),nn.ReLU())
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    @torch.jit.script_method
    def forward(self, x):
        ##################################################################
        # TODO 1.1: Forward the discriminator assuming a batch of images
        # have been passed in. Make sure to sum across the image
        # dimensions after passing x through self.layers.
        ##################################################################
        x = self.layers(x)
        
        x = x.sum(dim=-1).sum(dim=-1)
        return self.dense(x)
        x = self.layers(x)
        
        x = x.sum(dim=-1).sum(dim=-1)
        return self.dense(x)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################
