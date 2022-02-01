import torch
import torch.nn as nn
from math import sqrt

class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.conv(x))
        
class vdsr(nn.Module):
    def __init__(self):
        super(vdsr, self).__init__()
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 18)
        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out,residual)
        return out


class vdsrd2(nn.Module):
    def __init__(self):
        super(vdsrd2, self).__init__()
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 18)
        self.input = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x, zg):
        residual = x
        x_1=torch.cat((x,zg),dim=1)
        out = self.relu(self.input(x_1))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out,residual)
        return out

    
    
class cfsr(nn.Module):
    def __init__(self):
        super(cfsr, self).__init__()
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 18)
        
        self.residual_layer_2 = self.make_layer(Conv_ReLU_Block, 18)
        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out,residual)
        
        
        return out
    
class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std,channels, sign=-1):
        super(MeanShift, self).__init__(channels, channels, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(channels).view(channels, channels, 1, 1)
        self.weight.data.div_(std.view(channels, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False    
    
class vdsr_dem(nn.Module):
    def __init__(self):
        super(vdsr_dem, self).__init__()
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 18)
        self.input = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        rgb_mean_dem=[0.05986051]
        rgb_std_dem = [1.0]
        self.sub_mean_dem = MeanShift(2228.3303, rgb_mean_dem, rgb_std_dem,1)  
        
        rgb_mean_pr=[0.00216697]
        rgb_std_pr = [1.0]
        self.sub_mean_pr =MeanShift(993.9646, rgb_mean_pr, rgb_std_pr,1)
        self.add_mean = MeanShift(993.9646, rgb_mean_pr, rgb_std_pr,1,1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)
    
    

    def forward(self, x,dem):
        dem=self.sub_mean_dem(dem)
        residual = self.sub_mean_pr(x)
        x_1=torch.cat((x,dem),dim=1)
        
        out = self.relu(self.input(x_1))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out,residual)
        out=self.add_mean(out)
        return out