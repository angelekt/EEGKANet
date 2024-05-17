import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F




class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)


class EEGNetv4(nn.Module):
    def __init__(
        self,
        in_chans,
        n_classes,
        final_conv_length="auto",
        pool_mode="mean",
        F1=8,
        D=2,
        F2=16,  # usually set to F1*D (?)
        kernel_length=64,
        third_kernel_size=(8, 4),
        drop_prob= 0.,#cfg['drop_out_rate'],
        samples = 257#,cfg['samples'] #513#257
    ):
        super(EEGNetv4, self).__init__()

        self.channels = in_chans ## eeg channels
        self.samples  = samples #257 ## eeg timsamples
        self.n_classes = n_classes
        self.final_conv_length = final_conv_length
        self.pool_mode = pool_mode
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.kernelLength = kernel_length
        self.third_kernel_size = third_kernel_size
        self.dropoutRate = drop_prob
        

        self.blocks = self.conv_block()
        self.blockOutputSize = self.CalculateOutSize(self.blocks, self.channels, self.samples)
        self.clsBlock = self.ClassifierBlock(self.F2 * self.blockOutputSize[1], n_classes)


    def conv_block(self):
        self.block1 = nn.Sequential(
            ## conv2d temporal
            nn.Conv2d(1, self.F1, (1, self.kernelLength),stride=1, bias=False, padding=(0, self.kernelLength // 2),), ## in_channels = 1, out_channels = 16, kernel_size = (1,64)
            ## batchNorm 1 "bnorm_temporal"
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3), ## num_features = 16
            # Depthwise Conv2d 1 "conv_spatial"
            Conv2dWithConstraint(
                self.F1,
                self.F1 * self.D,
                (self.channels, 1),
                max_norm=1,
                stride=1,
                bias=False,
                groups=self.F1,
                padding=(0, 0),
            ),
            
            ## batchnorm 2
            nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            ## activation 1 ELU()
            nn.ELU(),
            ## AveragePool2d 1
            nn.AvgPool2d((1, 4), stride=(1, 4)),
            ## Dropout
            nn.Dropout(p=self.dropoutRate)
            )
        self.block2 = nn.Sequential(
            ### separable conv2d 1 "conv_separable_depth"
            nn.Conv2d(
                self.F1 * self.D,
                self.F1 * self.D,
                (1, 16),
                stride=1,
                bias=False,
                groups=self.F1 * self.D,
                padding=(0, 16 // 2),
            ),
            ##### "conv_separable_point",
            nn.Conv2d(
                self.F1 * self.D,
                self.F2,
                (1, 1),
                stride=1,
                bias=False,
                padding=(0, 0),
            ),
            ## Batchnorm1
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
            ## activation ELU
            nn.ELU(),
            ## AveragePool2d 2
            nn.AvgPool2d((1, 8), stride=(1, 8)),
            ## Dropout 2
            nn.Dropout(p=self.dropoutRate)
        )
        return nn.Sequential(self.block1,self.block2)

    def ClassifierBlock(self, inputSize, n_classes):
        # return nn.Sequential(
        #     nn.Linear(inputSize, n_classes, bias=False),
        #     nn.Softmax(dim=1))
        return nn.Sequential(
            nn.Linear(inputSize, n_classes, bias=False))
    
    def CalculateOutSize(self, model, channels, samples):
        '''
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        '''
        data = torch.rand(1, 1, channels, samples)
        model.eval()
        out = model(data).shape
        return out[2:]
    
    def forward(self,x):
        x = self.blocks(x)
        x = x.view(x.size()[0], -1)  # Flatten
        scores = self.clsBlock(x)
        return scores

from torch.nn.modules.module import _addindent

def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(paramtest_accuracy_best_model_models)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    return tmpstr