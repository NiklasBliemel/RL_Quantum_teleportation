import torch
import torch.nn as nn
import random
from IPython.display import clear_output
import matplotlib.pyplot as plt


class Tile:
    def __init__(self, grid_size, N_grids, left_bound=-1, right_bound=1):
        self.max_index = grid_size - 1
        self.N_grids = N_grids
        self.d = 1 / N_grids
        self.scale = (right_bound - left_bound)
        self.shift = -left_bound
        
    def __call__(self, x):
        out = (x + self.shift) / self.scale
        out = self.max_index * out.unsqueeze(0) + (self.d * torch.arange(self.N_grids).unsqueeze(-1))
        out = torch.trunc(out).int()
        out[out > self.max_index] = self.max_index
        out[out < 0] = 0
        return torch.matmul(out, (self.max_index + 1)**torch.arange(len(x)).int())


# test_tile = Tile(5, 8)

# def f(x,y):
#     plt.plot(x, y, marker="o")
#     plt.grid(True)
#     plt.xticks(torch.linspace(-1,1,5))
#     plt.yticks(torch.linspace(-1,1,5))
#     plt.title(str(test_tile(torch.tensor([x, y]))))
#     plt.show()

# ipywidgets.interact(f, x=(-1.0,1.0, 0.01), y=(-1.0,1.0, 0.01))


class SemiGradSarsa:
    def __init__(self, eps = 0.1, gamma = 1., alpha = 0.1):
        self.eps = eps 
        self.gamma = gamma
        self.alpha = alpha
        self.log = []

    def get_log(self):
        return self.log

    def set_log(self, log=None):
        if log is None:
            self.log = []
        else:
            self.log = log

    def update_weights(self, scale, q_net):
        for w in q_net.parameters():
            with torch.no_grad():
                w += scale * w.grad

    def eps_greedy(self, eps, q_values):
        if random.random() < eps or torch.abs(torch.std(q_values)) < 1e-9:
            A = random.randint(0, len(q_values) - 1)
            return A
        else:
            return torch.argmax(q_values)

    def plot_log(self):
        plt.plot(self.log)
        plt.xlabel("Episodes")
        plt.ylabel("Steps")
        # plt.yscale("log")
        plt.show()

    def run(self, env, q_net, max_episode, max_steps):
        try:
            for ep in range(max_episode):
                S = env.setup()
                q_values = q_net(S)
                A = self.eps_greedy(self.eps, q_values)
                q_net.zero_grad()
                q_values[A].backward()
                for step in range(max_steps):
                    
                    S_new, R, goal_reached = env.step(S, A)
                    
                    if goal_reached:
                        scale = self.alpha * (R - q_values[A])
                        self.update_weights(scale, q_net)
                        break
            
                    q_values_new = q_net(S_new)
                    A_new = self.eps_greedy(self.eps, q_values_new)
                    scale = self.alpha * (R + self.gamma * q_values_new[A_new] - q_values[A])
                    self.update_weights(scale, q_net)
        
                    A = A_new
                    if isinstance(S, torch.Tensor):
                        S = S_new.detach().clone()
                    else:
                        S = S_new * 1
                    q_net.zero_grad()
                    q_values = q_net(S)
                    q_values[A].backward()
                    
                self.log.append(step+1)
                if ep%10==0:
                    clear_output(wait=True)
                    self.plot_log()
        except KeyboardInterrupt:
            clear_output(wait=True)
            self.plot_log()

