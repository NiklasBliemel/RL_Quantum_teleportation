from QbitSim import *
from torch import nn
import os
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold, EvalCallback, StopTrainingOnNoModelImprovement
from IPython.display import clear_output


class Experiment:

    message = """""
    set_max_steps(max_steps)
    set_N_qbits(N_qbits)
    set_allowed_gates(allowed_gates)
    set_action_repetitions(action_repetitions)
    set_max_cnots(max_cnots)
    set_policy_size(pi=list[int], vf=list[int])
    save(path)
    load(path)
    train(frames)
    test(path=None)
    """""

    def __init__(self, exp_name):
        self.exp_name = exp_name
        self.max_steps = 12
        self.N_qbits = 3
        self.allowed_gates = "H X Z M CX"
        self.action_repetitions = 1
        self.max_cnots = 2
        self.policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
        self._make_env()

    def set_max_steps(self, max_steps):
        self.max_steps = max_steps
        self._make_env()

    def set_N_qbits(self, N_qbits):
        self.N_qbits = N_qbits
        self._make_env()

    def set_allowed_gates(self, allowed_gates):
        self.allowed_gates = allowed_gates
        self._make_env()

    def set_action_repetitions(self, action_repetitions):
        self.action_repetitions = action_repetitions
        self._make_env()

    def set_max_cnots(self, max_cnots):
        self.max_cnots = max_cnots
        self._make_env()

    def set_policy_size(self, pi: list[int], vf: list[int]):
        self.policy_kwargs = dict(net_arch=dict(pi=pi, vf=vf))
        self._make_env()

    def _make_env(self):
        self.env = QbitEnv(N_qbits=self.N_qbits, allowed_gates=self.allowed_gates, max_steps=self.max_steps,
                           action_repetition=self.action_repetitions, max_cnots=self.max_cnots)

        self.model = MaskablePPO("MlpPolicy", self.env, policy_kwargs=self.policy_kwargs, verbose=1)
        print(f"""
            exp_name = {self.exp_name}
            max_steps(max_steps) = {self.max_steps}
            N_qbits(N_qbits) = {self.N_qbits}
            allowed_gates(allowed_gates) = {self.allowed_gates}
            action_repetitions(action_repetitions) = {self.action_repetitions}
            max_cnots(max_cnots) = {self.max_cnots}
            policy_size(pi=list[int], vf=list[int]) = {self.policy_kwargs}
            """)

    def train(self, reward_threshold):
        callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold, verbose=1)

        eval_callback = EvalCallback(
            self.env,
            eval_freq=10_000,
            callback_on_new_best=callback_on_best,
            verbose=1,
            best_model_save_path=f"models/{self.exp_name}"
        )

        try:
            self.model.learn(total_timesteps=int(1e10), tb_log_name=self.exp_name, callback=eval_callback)
        except KeyboardInterrupt:
            self.model.learn(total_timesteps=int(1e3))

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = MaskablePPO.load(path)

    def test(self, path=None):
        test_model = MaskablePPO.load(path) if path is not None else self.model
        rew_mean, var = evaluate_policy(test_model, self.env, n_eval_episodes=100, reward_threshold=-20, warn=False)
        print(f"Mean reward = {rew_mean:.2f}")
        obs, _ = env.reset()
        for _ in range(max_steps):
            action_masks = get_action_masks(self.env)
            action, _states = test_model.predict(obs, action_masks=action_masks)
            obs, reward, terminated, truncated, info = self.env.step(action)
            print(info)
            if terminated:
                print("Success!")
                break
        if not terminated:
            print("Failure!")
