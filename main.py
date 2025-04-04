import QbitSim as qsim

if __name__ == '__main__':
    # Environment settings

    max_steps = 12
    fail_reward = -9
    N_qbits = 3
    max_cnots = 2

    env = qsim.QbitEnv(N_qbits=N_qbits, max_steps=max_steps, fail_reward=fail_reward, max_cnots=max_cnots)

    # training settings

    reward_threshold = -9
    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    model_name = "test"

    trainer = qsim.Trainer(env, reward_threshold, model_name, policy_kwargs=policy_kwargs)

    trainer.train()