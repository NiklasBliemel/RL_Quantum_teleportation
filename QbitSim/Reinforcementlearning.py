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

    def run(self, env, q_net, max_episode, max_steps, plot_range=None):
        info_log = []
        try:
            for ep in range(max_episode):
                ep_log = {"episode": ep, "win": False, "actions": []}

                S, info = env.reset()
                ep_log["reset"] = info

                q_values = q_net(S)
                A = self.eps_greedy(self.eps, q_values + env.get_action_mask())
                q_net.zero_grad()
                q_values[A].backward()
                for step in range(max_steps):

                    S_new, R, goal_reached, trunc, info = env.step(A)
                    ep_log["actions"].append(info)

                    if goal_reached:
                        ep_log["win"] = goal_reached
                        scale = self.alpha * (R - q_values[A])
                        self.update_weights(scale, q_net)
                        break

                    q_values_new = q_net(S_new)
                    A_new = self.eps_greedy(self.eps, q_values_new + env.get_action_mask())
                    scale = self.alpha * (R + self.gamma * q_values_new[A_new] - q_values[A])
                    self.update_weights(scale, q_net)

                    A = A_new

                    S = S_new.clone().detach() if torch.is_tensor(S_new) else S_new * 1

                    q_net.zero_grad()
                    q_values = q_net(S)
                    q_values[A].backward()

                self.log.append(step + 1)
                if ep % 10 == 0:
                    clear_output(wait=True)
                    self.plot_log(plot_range=plot_range)

                ep_log["steps"] = step + 1
                info_log.append(ep_log)

        except KeyboardInterrupt:
            clear_output(wait=True)
            self.plot_log(plot_range=plot_range)

        return info_log

    def plot_log(self, max_plot_length=1000, plot_range=None):
        L = len(self.log)
        if L < max_plot_length:
            x_range = torch.arange(1, L + 1)
        else:
            x_range = torch.arange(L - max_plot_length + 1, L + 1)
        plt.plot(x_range, self.log[-max_plot_length:])
        if plot_range is not None:
            plt.ylim(plot_range)
        plt.grid(True)
        plt.xlabel("Episodes")
        plt.ylabel("Total Reward")
        plt.show()

    def setting(self, eps=None, gamma=None, alpha=None):
        self.eps = eps if eps is not None else self.eps
        self.gamma = gamma if gamma is not None else self.gamma
        self.alpha = alpha if alpha is not None else self.alpha

    def get_log(self):
        return self.log

    def set_log(self, log=None):
        if log is None:
            self.log = []
        else:
            self.log = log
