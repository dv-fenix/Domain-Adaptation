import torch
from torch.autograd import Function
import torch.nn as nn

class GradientReversalLayer(Function):
  ''' Custom Layer to reverse the direction of gradients
  forward pass : returns input
  backward pass : gradient reversal
  '''
  @staticmethod
  def forward(ctx, input, completion_lambda: float=1.0):
      # Store for backward
      ctx.completion_lambda = (completion_lambda)
      # forward pass returns input
      return input.view_as(input)

  @staticmethod
  def backward(ctx, grad_output):
      completion_lambda = ctx.completion_lambda
      # Apply reversal
      output = grad_output.neg() * completion_lambda

      return output, None


class DANN(nn.Module):
    ''' Deep Adversairal Neural Network - [Convolutional]
    Label Channel : LeNet -> labels
    Domain Channel : Built over deep feature space with a gradient reversal during backprop -> domain labels
    '''

    def __init__(self, image_dim: int = 28):
        # constructor
        super(DANN, self).__init__()
        self.deep_feature_space = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5),
            nn.BatchNorm2d(32), nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.Conv2d(32, 48, kernel_size=5),
            nn.BatchNorm2d(48), nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(True),
        )

        self.label_channel = nn.Sequential(
            nn.Linear(48 * 4 * 4, 100), nn.BatchNorm1d(100), nn.Dropout(),
            nn.ReLU(True),
            nn.Linear(100, 100), nn.Dropout(),
            nn.ReLU(True),
            nn.Linear(100, 10),
            nn.Softmax(dim=1)
        )

        self.domain_channel = nn.Sequential(
            nn.Linear(48 * 4 * 4, 100), nn.BatchNorm1d(100), nn.Dropout(),
            nn.ReLU(True),
            nn.Linear(100, 2),
            nn.Softmax(dim=1)
        )
        self.size = image_dim

    def forward(self, inputs, completion_lambda: float = 1.0):
        inputs = inputs.expand(-1, 3, self.size, self.size)
        # Build LeNet
        deep_feature = self.deep_feature_space(inputs)
        deep_feature = deep_feature.view(-1, 48 * 4 * 4)
        labels = self.label_channel(deep_feature)

        # Domain Classifier with Gradient Reversal
        gradient_reversal = GradientReversalLayer.apply(deep_feature, completion_lambda)
        domain_labels = self.domain_channel(gradient_reversal)

        return labels, domain_labels