import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
from dm_env import specs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gym_env
from replay_buffer_parallel import ReplayBufferStorageParallel, make_replay_loader
from agent.rover_nystrom_pointmaze_debug import RoverAgent


def _sample_actions(env):
    return np.asarray([env.single_action_space.sample() for _ in range(env.num_envs)])


def _time_steps(infos, mask=None):
    values = infos["time_step"]
    if mask is None:
        mask = np.ones(len(values), dtype=bool)
    return [values[i] if mask[i] else None for i in range(len(values))]


def check_env(task_name, obs_type, env_kwargs):
    for num_envs in (1, 2, 4):
        env = gym_env.make_async_vector_env(
            num_envs,
            123,
            task_name,
            obs_type,
            frame_stack=1,
            action_repeat=1,
            url=False,
            **env_kwargs,
        )
        try:
            obs, infos = env.reset(seed=[123 + i for i in range(num_envs)])
            assert obs.shape[0] == num_envs
            assert "time_step" in infos
            next_obs, rewards, terminated, truncated, infos = env.step(_sample_actions(env))
            assert next_obs.shape[0] == num_envs
            assert rewards.shape == (num_envs,)
            assert terminated.shape == (num_envs,)
            assert truncated.shape == (num_envs,)
            done = np.logical_or(terminated, truncated)
            if np.any(done):
                reset_obs, reset_infos = env.reset(options={"reset_mask": done.astype(np.bool_)})
                assert reset_obs.shape[0] == num_envs
                reset_steps = _time_steps(reset_infos, done)
                for env_id in np.flatnonzero(done):
                    assert reset_steps[env_id].first()
        finally:
            env.close()


def check_replay(task_name, obs_type, env_kwargs):
    env = gym_env.make_async_vector_env(
        2,
        321,
        task_name,
        obs_type,
        frame_stack=1,
        action_repeat=1,
        url=False,
        **env_kwargs,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        single_env = gym_env.make(task_name, obs_type, frame_stack=1, action_repeat=1, seed=321, url=False, **env_kwargs)
        obs_spec = gym_env.observation_spec(single_env)
        action_spec = gym_env.action_spec(single_env)
        data_specs = (
            obs_spec,
            action_spec,
            specs.Array((1,), np.float32, "reward"),
            specs.Array((1,), np.float32, "discount"),
        )
        storage = ReplayBufferStorageParallel(data_specs, tuple(), Path(tmpdir), num_envs=2)
        try:
            obs, infos = env.reset(seed=[321, 322])
            metas = [{}, {}]
            steps = _time_steps(infos)
            for env_id, time_step in enumerate(steps):
                storage.add(time_step, metas[env_id], env_id=env_id)

            collected = 0
            while len(storage) < 2 and collected < 1200:
                obs, rewards, terminated, truncated, infos = env.step(_sample_actions(env))
                done = np.logical_or(terminated, truncated)
                steps = _time_steps(infos)
                for env_id, time_step in enumerate(steps):
                    storage.add(time_step, metas[env_id], env_id=env_id)
                if np.any(done):
                    reset_obs, reset_infos = env.reset(options={"reset_mask": done.astype(np.bool_)})
                    reset_steps = _time_steps(reset_infos, done)
                    for env_id in np.flatnonzero(done):
                        storage.add(reset_steps[env_id], metas[env_id], env_id=env_id)
                collected += 2

            assert len(storage) > 0
            for fn in Path(tmpdir).glob("*.npz"):
                episode = np.load(fn)
                env_ids = episode["env_id"]
                assert np.all(env_ids == env_ids[0])

            loader = make_replay_loader(storage, 1000, 2, 0, False, 1, 0.99)
            if len(storage) > 0:
                next(iter(loader))
        finally:
            single_env.close()
            env.close()


def check_act_parallel_parity():
    agent = RoverAgent(
        name="smoke",
        obs_type="states",
        obs_shape=(2,),
        grayscale=False,
        action_shape=(3,),
        lr_actor=1e-3,
        discount=0.99,
        lambda_reg=0.0,
        batch_size=2,
        batch_size_actor=2,
        subsamples=2,
        nstep=1,
        use_tb=False,
        use_wandb=False,
        lr_T=1e-3,
        lr_encoder=1e-3,
        curl=False,
        embedding_sum_loss=False,
        hidden_dim=8,
        feature_dim=4,
        update_every_steps=1,
        update_actor_every_steps=1,
        pmd_steps=1,
        num_expl_steps=0,
        T_init_steps=0,
        total_train_steps=10,
        sink_schedule="0",
        epsilon_schedule="0",
        mode="l2",
        reward="intrinsic",
        pca_truncation=0,
        embeddings=True,
        device="cpu",
    )
    observations = np.asarray([[0.0, 0.0], [0.1, -0.1], [1.0, 0.5]], dtype=np.float32)
    metas = [agent.init_meta() for _ in range(observations.shape[0])]
    steps = np.asarray([10, 11, 12], dtype=np.int64)
    np.random.seed(7)
    looped = np.asarray([
        agent.act(observations[i], metas[i], int(steps[i]), eval_mode=True)
        for i in range(observations.shape[0])
    ])
    np.random.seed(7)
    batched = agent.act_parallel(observations, metas, steps, eval_mode=True)
    assert batched.shape == (observations.shape[0],)
    assert np.array_equal(looped, batched)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", default="CartPole-v1")
    parser.add_argument("--obs-type", default="states")
    args, unknown = parser.parse_known_args()
    env_kwargs = {}
    for item in unknown:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        env_kwargs[key.lstrip("-")] = value

    check_env(args.task_name, args.obs_type, env_kwargs)
    check_replay(args.task_name, args.obs_type, env_kwargs)
    check_act_parallel_parity()
    print("parallel env smoke tests passed")


if __name__ == "__main__":
    main()
