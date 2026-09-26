import torch
from torch import nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Custom feature extractor class to flatten (2,5,5,5) state tensor 
class Custom_Flatten(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 250):
        super().__init__(observation_space, features_dim)
        # torch modules to flatten (2,5,5,5) numpy array
        self.flatten = nn.Flatten(start_dim=1,end_dim=-1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        out = self.flatten(observations)
        return out

class IMPALA_CNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        self.flatten = nn.Flatten(start_dim=1,end_dim=2)
        self.conv1 = nn.Conv2d(in_channels=10,out_channels=20,kernel_size=3,stride=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=40,kernel_size=3,stride=1)
        self.relu2 = nn.ReLU()
        self.flatten_after_cnn = nn.Flatten()
        self.linear = nn.Linear(40,128)
        self.relu3 = nn.ReLU()
        self.lstm = nn.LSTM(128,features_dim)

    def forward(self, observations) -> torch.Tensor:
        x = torch.Tensor(observations)
        # flatten to get from tensor of shape (1,2,5,5,5) to (1,10,5,5)
        x = self.flatten(x)
        # IMPALA CNN block (1,10,5,5) -> (1,20,3,3) 
        x = self.conv1(x)
        x = self.relu1(x)
        # (1,20,3,3) -> (1,40,1,1)
        x = self.conv2(x)
        x = self.relu2(x)
        # flatten to shape 128
        x = self.flatten_after_cnn(x)
        # fully-connected-layer 40 -> 128
        x = self.linear(x)
        x = self.relu3(x)
        # Long Short Term Memory 128 -> 128
        output, (hn,cn) = self.lstm(x)

        return output


class IMPALA_CNN_Large(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        self.flatten = nn.Flatten(start_dim=1,end_dim=2)
        self.impala_cnn_block1 = IMPALA_CNN_Block(num_input_channels=10,num_out_channels=20)
        self.impala_cnn_block2 = IMPALA_CNN_Block(num_input_channels=20,num_out_channels=40)
        self.impala_cnn_block3 = IMPALA_CNN_Block(num_input_channels=40, num_out_channels=40)

        self.relu1 = nn.ReLU()
        self.flatten_after_cnn = nn.Flatten()
        self.linear = nn.Linear(40*2*2,128)
        self.relu2 = nn.ReLU()
        self.lstm = nn.LSTM(128,features_dim)

    def forward(self, observations) -> torch.Tensor:
        x = torch.Tensor(observations)
        # flatten to get from tensor of shape (1,2,5,5,5) to (1,10,5,5)
        x = self.flatten(x)
        # IMPALA CNN block (1,10,5,5) -> (1,20,4,4) -> (1,40,3,3) -> (1,40,2,2)
        x = self.impala_cnn_block3(self.impala_cnn_block2(self.impala_cnn_block1(x)))
        x = self.relu1(x)
        # flatten to shape 128
        x = self.flatten_after_cnn(x)
        # fully-connected-layer 128 -> 128
        x = self.linear(x)
        x = self.relu2(x)
        # Long Short Term Memory 128 -> 128
        output, (hn,cn) = self.lstm(x)

        return output
    
class IMPALA_CNN_Block(torch.nn.Module):
    '''
    input: env_state needs to be in format (1,10,5,5), where the 1=batch size. and the (10,5,5) is the flattened
           version of the (2,5,5,5) observations
    '''

    def __init__(self, num_input_channels=10, num_out_channels=16):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=num_input_channels,out_channels=num_out_channels,kernel_size=3,stride=1,padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2,stride=1)

        self.residual_block1 = Residual_Block(num_input_channels=num_out_channels,num_out_channels=num_out_channels)
        self.residual_block2 = Residual_Block(num_input_channels=num_out_channels,num_out_channels=num_out_channels)

    def forward(self,input):
        x = self.conv1(input)
        x = self.max_pool(x)
        x = self.residual_block1(x)
        output = self.residual_block2(x)

        return output

class Residual_Block(torch.nn.Module):

    def __init__(self, num_input_channels=10, num_out_channels=16):
        super().__init__()

        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=num_input_channels,out_channels=num_out_channels,kernel_size=3,stride=1,padding=1)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=num_input_channels,out_channels=num_out_channels,kernel_size=3,stride=1,padding=1)

    def forward(self,residual_block_input):
        x = self.relu1(residual_block_input)
        x = self.conv1(x)
        x = self.relu2(x)
        x = self.conv2(x)

        return x + residual_block_input


        


