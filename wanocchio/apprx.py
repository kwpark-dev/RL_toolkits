import torch
import torch.nn as nn



def fc_apprx(in_dim, out_dim, N, end_act=nn.Sigmoid):
    # pre-defined fc layer architecture
    # make sure that the number of data points is more than 1 (batchnorm)
    
    front = [nn.Linear(in_dim, 64),
             nn.ReLU()]
    
    end = [nn.Linear(64, out_dim),
           end_act()]
    # should think about layernorm for better training
    hidden = [nn.Linear(64, 64),
            #   nn.BatchNorm1d(64),
              nn.ReLU()]*N
    
    func = front + hidden + end
    
    
    return nn.Sequential(*func)



def conv_apprx(state_dim, action_dim):
    # pre-defined conv layer architecture
    pass



def custom_apprx(config):
    # user-defined architecture through config
    
    pass



# if __name__ == '__main__':
#     data = torch.rand(3, 10)
    
#     func = fc_apprx(10, 5, 3)
    
#     res = func(data)
#     print(res.shape)