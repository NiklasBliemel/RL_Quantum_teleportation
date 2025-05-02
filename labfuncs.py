from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
import os
import numpy as np
import QbitSim as qsim

N_qbits = 3
max_steps = 12
fail_reward = -9
max_cnots = 2

env = qsim.QbitEnv(N_qbits=N_qbits, max_steps=max_steps, fail_reward=fail_reward, max_cnots=max_cnots)


def test(model_name):
    path = f"models/{model_name}/best_model.zip"
    test_model = MaskablePPO.load(path)
    obs, _ = env.reset()
    for _ in range(max_steps):
        # Retrieve current action mask
        action_masks = get_action_masks(env)
        action, _states = test_model.predict(obs, action_masks=action_masks)
        obs, reward, terminated, truncated, info = env.step(action)
        print(info)
        if terminated:
            print("Success!")
            break
    if not terminated:
        print("Failure!")


def last_timestep(model_name):
    path = f"models/{model_name}/log.npz"
    log = np.load(path)
    return log['timesteps'][-1]


def best_val(model_name):
    path = f"models/{model_name}/log.npz"
    log = np.load(path)
    return np.max(log['results'])
