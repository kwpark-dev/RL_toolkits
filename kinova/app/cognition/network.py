import torch
import torch.nn as nn



class ResidualEncoder(nn.Module):
    def __init__(self, input_channel, output_dim, is_multi_head=False):
        super().__init__()

        self.input = input_channel
        self.output = output_dim
        self.is_multi_head = is_multi_head

        self.maxpool = nn.MaxPool2d(2, 2)
        self.gapool = nn.AvgPool2d(32, 32)
        self.conv_1x1_64 = self._conv_1x1(32, 64)
        self.conv_1x1_32 = self._conv_1x1(64, 128)

        self.module_128 = self._build_module(128, 3, 32, 3, 1, 1)
        self.module_64 = self._build_module(64, 32, 64, 3, 1, 1)
        self.module_32 = self._build_module(32, 64, 128, 3, 1, 1)

        self.fc = nn.Linear(128, 2*output_dim)

        if self.is_multi_head:
            self.fc2 = nn.Linear(128, 2)


    def _build_module(self, size, cin, cout, k, s, p):
        module = nn.Sequential(
                    nn.Conv2d(cin, cout, k, stride=s, padding=p),
                    nn.LayerNorm([cout, size, size]),
                    nn.GELU())

        return module


    def _conv_1x1(self, cin, cout):
        module = nn.Conv2d(cin, cout, 1)

        return module


    def forward(self, x):
        y = self.module_128(x)
        
        x = self.maxpool(y)
        y = self.module_64(x)
        y = y + self.conv_1x1_64(x)

        x = self.maxpool(y)
        y = self.module_32(x)
        feature = y + self.conv_1x1_32(x)
        
        x = self.gapool(feature) 
        y = x.squeeze()

        if self.is_multi_head:
            z = self.fc2(y)
            y = self.fc(y)

            return feature, x, z.unsqueeze(dim=0), y.unsqueeze(dim=0)

        else:
            y = self.fc(y)

            return feature, x, y.unsqueeze(dim=0)



class ValueEncoder(ResidualEncoder):
    def __init__(self, input_channel, output_dim, is_multi_head=False):
        super().__init__(input_channel, output_dim, is_multi_head)

        self.fc = nn.Linear(128, output_dim)




if __name__ == "__main__":
    test_img = torch.rand(1, 3, 128, 128)
    
    actor = ResidualEncoder(3, 7, True)
    critic = ValueEncoder(3, 1)
    
    featac, wac, util, res = actor(test_img)
    featcr, wcr, scr = critic(test_img)

    print(res.shape, util.shape)
    print(scr, wcr.shape, featcr.shape)


