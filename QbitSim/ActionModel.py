import torch.nn as nn
import torch


class ActionModel(nn.Module):
    def __init__(self, max_steps, env):
        super(ActionModel, self).__init__()
        n_actions = env.action_space.n
        self.w = nn.Linear(max_steps * n_actions * 3, n_actions, dtype=torch.float32, bias=False)
        self.start = nn.Parameter(torch.zeros(n_actions, dtype=torch.float32, requires_grad=True))

    def forward(self, S):
        if not isinstance(S, torch.Tensor):
            out = torch.tensor(S)
        else:
            out = S.clone().detach()
        if torch.all(out == 0.):
            return self.start
        else:
            return self.w(out)