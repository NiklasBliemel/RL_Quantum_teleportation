import torch.nn as nn
import torch


class ActionModel(nn.Module):
    def __init__(self, max_steps, env):
        super(ActionModel, self).__init__()
        n_actions = env.action_space.n
        self.w = nn.Parameter(
            torch.zeros(max_steps, n_actions * 3, n_actions, dtype=torch.float32, requires_grad=True))
        self.start = nn.Parameter(torch.zeros(n_actions, dtype=torch.float32, requires_grad=True))

    def forward(self, S):
        out = S.clone().detach() if torch.is_tensor(S) else torch.tensor(S)
        out = out[out != -1]
        return self.w[torch.arange(len(out)), out].sum(axis=0) + self.start


class DynamicActionModel(nn.Module):
    def __init__(self, max_steps, env):
        super(DynamicActionModel, self).__init__()

        self.env = env
        self.cnots = env.get_cnot_index()

        n_actions = env.action_space.n
        self.w = nn.Parameter(torch.zeros(max_steps * n_actions * 3, n_actions, dtype=torch.float32, requires_grad=True))
        self.start = nn.Parameter(torch.zeros(n_actions, dtype=torch.float32, requires_grad=True))

    def forward(self, S):
        out = S.clone().detach() if torch.is_tensor(S) else torch.tensor(S)

        if torch.all(out == 0.):
            out = self.start
        else:
            out = out @ self.w

        if self.env.out_of_cnots():
            with torch.no_grad():
                out[self.cnots] = -torch.inf

        return out
