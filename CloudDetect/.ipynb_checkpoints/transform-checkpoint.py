import numpy as np
from torchvision import transforms
import torch

class PointCloudSample(object):
    def __init__(self, output_sample_size):
        self.output_sample_size = output_sample_size

    def __call__(self,data):
        sample_idx = np.random.choice(len(data),self.output_sample_size)
        return data[sample_idx]

class Normalise(object):
    def __init__(self, how):
        self.how = how
        
    def __call__(self, data):
        assert len(data.shape)==2

        if self.how == 'max':
        
            norm_data = data - np.mean(data, axis=0) 
            norm_data /= np.max(np.linalg.norm(norm_data, axis=1))
    
            return  norm_data

        else:
            raise NotImplementedError(f'{self.how} not implemented yet!')
            

class Tensor(object):
    def __call__(self, data):
        return torch.tensor(data)