import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class ActorCritic(torch.nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()

        # FC and embedding
        self.fc1 = nn.Linear(8192, 256)
        self.embedding_layer = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        # actor-critic
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, 4)

        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

    def forward(self, state_input, target_input):
        # siamese
        h1_s = self.relu1(self.fc1(state_input))
        h1_t = self.relu1(self.fc1(target_input))

        # concat state and state_target
        input_embedding = torch.cat((h1_s, h1_t), 1)  # input_embedding:[1, 1024]
        h2_e = self.relu2(self.embedding_layer(input_embedding))  # h2_e:[1, 512]

        h3 = self.relu2(self.fc2(h2_e))  # h3:[1, 512]
        value_output = self.critic_linear(h3)
        logit_output = self.actor_linear(h3)

        return value_output, logit_output
