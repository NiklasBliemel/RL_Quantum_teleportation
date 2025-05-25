from typing import List

import numpy as np

import torch

import QbitSim as qsim

from labfuncs import best_val

if __name__ == '__main__':
    # Environment settings
    N_qbits = 3
    max_steps = 12
    fail_reward = -5
    max_cnots = 2

    # model settings
    model_repetition = 10
    model_size = 256
    learning_rate = 0.0003
    ent_coef = 0.0
    reward_threshold = -10

    # initialize environment and trainer
    env = qsim.QbitEnv(N_qbits=N_qbits, max_steps=max_steps, fail_reward=fail_reward, max_cnots=max_cnots)
    trainer = qsim.Trainer(env, reward_threshold)

    # initialize and train new models
    for i in range(model_repetition):
        trainer.set_reward_threshold(-14)
        model_name = f"mlp_{i}_({model_size}_{learning_rate}_{ent_coef})"
        net_arch = dict(pi=[model_size, model_size], vf=[model_size, model_size])
        trainer.new_model(model_name, learning_rate, ent_coef, dict(net_arch=net_arch), force=True, verbose=0)
        trainer.train()
        if best_val(model_name) > -14:
            trainer.model.save(f"models_14/{model_name}")
            trainer.set_reward_threshold(-12)
            trainer.train()
            trainer.model.save(f"models_12/{model_name}")
            trainer.set_reward_threshold(-10)
            trainer.train()
