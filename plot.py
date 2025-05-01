from labfuncs import *
import matplotlib.pyplot as plt

models = []
for name in os.listdir("logs"):
    if best_val(name) > -10:
        models.append({"name": name, "data": np.load("logs/" + name)})
        print(name)
        print(best_val(name))
        print()

points = ["x", "o", "v"]


N=56
for i, model in enumerate(models):
    timesteps = model["data"]["timesteps"]
    results = np.mean(model["data"]["results"], axis=1)
    plt.plot(timesteps[N:], results[N:], linestyle="", marker=points[i], markersize=5, label=model["name"])
plt.ylim(-16,-6)
plt.xlabel("Timesteps")
plt.ylabel("Reward")
plt.hlines(y=-10, xmin=timesteps[56], xmax=7e5, colors='r', linestyles='--', label="Reward Threshold")
plt.legend().set_draggable(True)
plt.show()
