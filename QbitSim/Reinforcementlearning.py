import torch
import random
from IPython.display import clear_output
import matplotlib.pyplot as plt


class SemiGradSarsa:
    def __init__(self, eps=0.1, gamma=0.9, alpha=0.1):
        self.eps = eps
        self.gamma = gamma
        self.alpha = alpha
        self.log = []

    def setting(self, eps=0.1, gamma=1., alpha=0.1):
        self.eps = eps
        self.gamma = gamma
        self.alpha = alpha

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
                if w.grad is not None:
                    w += scale * w.grad

    def eps_greedy(self, eps, q_values):
        if random.random() < eps or torch.abs(torch.std(q_values[q_values != -torch.inf])) < 1e-9:
            w_list = [0 if q_val == -torch.inf else 1 for q_val in q_values]
            A = random.choices(list(range(len(q_values))), weights=w_list)[0]
            return A
        else:
            return torch.argmax(q_values)

    def plot_log(self, max_plot_length=1000, plot_range=None):
        L = len(self.log)
        if L < max_plot_length:
            x_range = torch.arange(1, L + 1)
        else:
            x_range = torch.arange(L - max_plot_length + 1, L + 1)
        plt.plot(x_range, self.log[-max_plot_length:])
        if plot_range is not None:
            plt.ylim(plot_range)
        plt.xlabel("Episodes")
        plt.ylabel("Steps")
        plt.show()

    def run(self, env, q_net, max_episode, max_steps, catch_wins=False, catch_all=False, plot_range=None):
        wins = []
        try:
            for ep in range(max_episode):
                action_log = []

                S, info = env.reset()
                q_values = q_net(S)
                A = self.eps_greedy(self.eps, q_values)
                q_net.zero_grad()
                q_values[A].backward()
                for step in range(max_steps):

                    S_new, R, goal_reached, trunc, info = env.step(A)

                    action_log.append(info)

                    if goal_reached:
                        if catch_wins and not catch_all:
                            wins.append(action_log)

                        scale = self.alpha * (R - q_values[A])
                        self.update_weights(scale, q_net)
                        break

                    q_values_new = q_net(S_new)
                    A_new = self.eps_greedy(self.eps, q_values_new)
                    scale = self.alpha * (R + self.gamma * q_values_new[A_new] - q_values[A])
                    self.update_weights(scale, q_net)

                    A = A_new
                    
                    if isinstance(S, torch.Tensor):
                        S = S_new.clone().detach()
                    else:
                        S = S_new * 1
                        
                    q_net.zero_grad()
                    q_values = q_net(S)
                    q_values[A].backward()

                self.log.append(step + 1)
                if ep % 10 == 0:
                    clear_output(wait=True)
                    self.plot_log(plot_range=plot_range)
                if catch_all:
                    wins.append(action_log)
        except KeyboardInterrupt:
            clear_output(wait=True)
            self.plot_log(plot_range=plot_range)

        if catch_wins or catch_all:
            return wins
