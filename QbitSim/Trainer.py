from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from time import time_ns
import os


class Trainer:
    def __init__(self, env, reward_threshold):
        self.env = Monitor(env)
        self.model = None

        self.callback_on_best = StopTrainingOnRewardThreshold(
            reward_threshold=reward_threshold,
            verbose=1
        )
        self.eval_callback = MaskableEvalCallback(
            self.env,
            n_eval_episodes=20,
            eval_freq=5000,
            callback_on_new_best=self.callback_on_best,
            verbose=1
        )

    def load_model(self, model_name):
        self.reset_callback()
        if model_name in os.listdir("models"):
            self.model = MaskablePPO.load(os.path.join("models", model_name, "best_model"), env=self.env)
            self.model.verbose = 0
            self.eval_callback.best_model_save_path = os.path.join("models", model_name)
            self.eval_callback.log_path = f"models/{model_name}/log"
            print(f"{model_name} initialized!")
        else:
            print("Model does not exist")

    def new_model(self, model_name, learning_rate=0.003, ent_coef=0.0, policy_kwargs=None, force=False, verbose=0):
        if model_name in os.listdir("models"):
            if force:
                var = "y"
            else:
                var = input("Model already exists, do you want to overwrite it? (y/n): ")

            if var == "n":
                self.load_model(model_name)
                return
        else:
            os.mkdir(f"models/{model_name}")
        self.reset_callback()
        self.model = MaskablePPO("MlpPolicy", self.env, learning_rate=learning_rate, ent_coef=ent_coef,
                                 policy_kwargs=policy_kwargs, verbose=verbose)
        self.eval_callback.best_model_save_path = os.path.join("models", model_name)
        self.eval_callback.log_path = f"models/{model_name}/log"
        self.model.save(os.path.join("models", model_name, "best_model"))
        print(f"{model_name} initialized!")

    def reset_callback(self):
        self.eval_callback = MaskableEvalCallback(
            self.env,
            n_eval_episodes=20,
            eval_freq=5000,
            callback_on_new_best=self.callback_on_best,
            verbose=1
        )

    def train(self):
        assert self.model is not None, "Model does not exist"
        try:
            self.model.learn(1e6, callback=self.eval_callback)
        except KeyboardInterrupt:
            self.model.learn(1000, callback=self.eval_callback)

    def performance_test(self):
        times = []
        for i in range(10):
            print(f"Test {i}")
            start = time_ns()
            self.model.learn(1024)
            times.append(time_ns() - start)
        total = sum(times) / 10
        print(f"Mean time: {total * 1e-6:.3f}ms")

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
