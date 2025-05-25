import numpy as np

from labfuncs import *
import matplotlib.pyplot as plt

models = []
for name in os.listdir("models"):
    if best_val(name) > -10:
        models.append({"name": name, "data": np.load(f"models/{name}/log.npz")})

points = ["x", "o", "v", "+", ".", ".", "."]

for i, model in enumerate(models):
    if model["name"] != "mlp_2_(256_0.0003_0.0)":
        results = np.mean(model["data"]["results"], axis=1)
        timesteps = 5000 * np.arange(1, len(results) + 1)
        plt.plot(timesteps[timesteps >= 2e5], results[timesteps >= 2e5], linestyle="", marker=points[i], markersize=5, label=model["name"])
plt.ylim(-17, -5)
plt.xlabel("Timesteps")
plt.ylabel("Reward")
plt.hlines(y=-14, xmin=2e5, xmax=9e5, colors='r', linestyles='--', label=f"Reward Threshold")
plt.hlines(y=-12, xmin=2e5, xmax=9e5, colors='r', linestyles='--')
plt.hlines(y=-10, xmin=2e5, xmax=9e5, colors='r', linestyles='--')
plt.legend(loc='best')
plt.savefig('plot.pdf', format='pdf')
