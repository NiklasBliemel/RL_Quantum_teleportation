from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import os


class Trainer:
    def __init__(self, env, reward_threshold, model_name, policy_kwargs):
        self.env = Monitor(env)
        self.model = MaskablePPO("MlpPolicy", self.env, policy_kwargs=policy_kwargs, verbose=0)
        self.callback_on_best = StopTrainingOnRewardThreshold(
            reward_threshold=reward_threshold,
            verbose=1
        )
        self.eval_callback = EvalCallback(
            self.env,
            n_eval_episodes=100,
            eval_freq=5000,
            best_model_save_path=os.path.join("models", model_name),
            log_path=os.path.join("models", model_name),
            callback_on_new_best=self.callback_on_best,
            verbose=1
        )

    def train(self):
        try:
            self.model.learn(1e10, callback=self.eval_callback)
        except KeyboardInterrupt:
            self.model.learn(1000, callback=self.eval_callback)

    def set_reward_threshold(self, reward_threshold):
        self.callback_on_best = StopTrainingOnRewardThreshold(
            reward_threshold=reward_threshold,
            vebose=1
        )
        self.eval_callback = EvalCallback(
            self.env,
            n_eval_episodes=100,
            eval_freq=1000,
            best_model_save_path=os.path.join("models", model_name),
            log_path=os.path.join("models", model_name + "_log"),
            callback_on_new_best=self.callback_on_best,
            verbose=1
        )
