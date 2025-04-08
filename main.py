from typing import List

import numpy as np

import QbitSim as qsim

if __name__ == '__main__':
    # Environment settings

    N_qbits = 3
    max_steps = 12
    fail_reward = -5
    max_cnots = 2
    reward_threshold = -10

    env = qsim.QbitEnv(N_qbits=N_qbits, max_steps=max_steps, fail_reward=fail_reward, max_cnots=max_cnots)
    trainer = qsim.Trainer(env, reward_threshold)

    # model settings

    model_repetition = 5
    model_sizes = [256, 512]
    default_model_size = 256
    learning_rates = [0.0001, 0.0007, 0.001]
    default_learning_rate = 0.0003
    ent_coefs = [0.001, 0.01, 0.1]
    default_ent_coef = 0.0

    best_convergence = 1e6

    for learning_rate in learning_rates:
        for i in range(model_repetition):
            model_name = f"mlp_{i}_({default_model_size}_{learning_rate}_{default_ent_coef})"
            policy_kwargs = dict(net_arch=dict(pi=[default_model_size, default_model_size], vf=[default_model_size, default_model_size]))
            trainer.new_model(model_name, learning_rate, default_ent_coef, policy_kwargs, force=True, verbose=0)
            trainer.train()
            log = np.load(f"logs/{model_name}.npz")
            last_timestep = log['timesteps'][-1]
            if last_timestep < best_convergence:
                best_convergence = last_timestep
                default_learning_rate = learning_rate

    for ent_coef in ent_coefs:
        for i in range(model_repetition):
            model_name = f"mlp_{i}_({default_model_size}_{default_learning_rate}_{ent_coef})"
            policy_kwargs = dict(net_arch=dict(pi=[default_model_size, default_model_size], vf=[default_model_size, default_model_size]))
            trainer.new_model(model_name, default_learning_rate, ent_coef, policy_kwargs, force=True, verbose=0)
            trainer.train()
            log = np.load(f"logs/{model_name}.npz")
            last_timestep = log['timesteps'][-1]
            if last_timestep < best_convergence:
                best_convergence = last_timestep
                default_ent_coef = ent_coef

    for model_size in model_sizes:
        for i in range(model_repetition):
            model_name = f"mlp_{i}_({model_size}_{default_learning_rate}_{default_ent_coef})"
            policy_kwargs = dict(net_arch=dict(pi=[model_size, model_size], vf=[model_size, model_size]))
            trainer.new_model(model_name, default_learning_rate, default_ent_coef, policy_kwargs, force=True, verbose=0)
            trainer.train()
            log = np.load(f"logs/{model_name}.npz")
            last_timestep = log['timesteps'][-1]
            if last_timestep < best_convergence:
                best_convergence = last_timestep
                default_model_size = model_size

    result = open('result.txt', 'w')
    result.write(f"best model size = {default_model_sizef}\n"
                 f"best learning rate = {default_learning_rate}\n"
                 f"best ent coef = {default_ent_coef}\n")
    result.close()
