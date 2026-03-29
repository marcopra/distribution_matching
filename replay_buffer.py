import datetime
import io
import random
import traceback
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open('wb') as f:
            f.write(bs.read())


def load_episode(fn):
    with fn.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


class ReplayBufferStorage:
    def __init__(self, data_specs, meta_specs, replay_dir):
        self._data_specs = data_specs
        self._meta_specs = meta_specs
        self._replay_dir = replay_dir
        replay_dir.mkdir(exist_ok=True)
        self._current_episode = defaultdict(list)
        self._preload()

    def __len__(self):
        return self._num_transitions

    def add(self, time_step, meta):
        for key, value in meta.items():
            self._current_episode[key].append(value)
        for spec in self._data_specs:
            value = time_step[spec.name]
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            assert spec.shape == value.shape and spec.dtype == value.dtype
            self._current_episode[spec.name].append(value)
        if time_step.last():
            episode = dict()
            for spec in self._data_specs:
                value = self._current_episode[spec.name]
                episode[spec.name] = np.array(value, spec.dtype)
            for spec in self._meta_specs:
                value = self._current_episode[spec.name]
                episode[spec.name] = np.array(value, spec.dtype)
            self._current_episode = defaultdict(list)
            self._store_episode(episode)

    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._replay_dir.glob('*.npz'):
            _, _, eps_len = fn.stem.split('_')
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def _store_episode(self, episode):
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        eps_fn = f'{ts}_{eps_idx}_{eps_len}.npz'
        save_episode(episode, self._replay_dir / eps_fn)


class ReplayBuffer(IterableDataset):
    def __init__(self, storage, max_size, num_workers, nstep, discount,
                 fetch_every, save_snapshot, first_transition=False, batch_size=None):
        self._storage = storage
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._nstep = nstep
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot
        self._first_transition = first_transition
        self._batch_size = batch_size
        self._transition_cache = dict()
        self._all_data_cache = None
        self._all_data_cache_key = None

    def _invalidate_transition_views(self):
        self._all_data_cache = None
        self._all_data_cache_key = None

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _store_episode(self, eps_fn):
        try:
            episode = load_episode(eps_fn)
        except:
            return False
        eps_len = episode_len(episode)
        while eps_len + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= episode_len(early_eps)
            self._transition_cache.pop(early_eps_fn, None)
            early_eps_fn.unlink(missing_ok=True)
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len
        self._invalidate_transition_views()

        if not self._save_snapshot:
            eps_fn.unlink(missing_ok=True)
        return True

    def _try_fetch(self, force=False):
        if not force and self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._storage._replay_dir.glob('*.npz'), reverse=True)
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split('_')[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._episodes.keys():
                break
            if fetched_size + eps_len > self._max_size:
                break
            fetched_size += eps_len
            if not self._store_episode(eps_fn):
                break

    def _sample(self, first=False):
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1) + 1
        if first:
            idx = 1
        meta = []
        for spec in self._storage._meta_specs:
            meta.append(episode[spec.name][idx - 1])
        obs = episode['observation'][idx - 1]
        action = episode['action'][idx]
        next_obs = episode['observation'][idx + self._nstep - 1]
        reward = np.zeros_like(episode['reward'][idx])
        discount = np.ones_like(episode['discount'][idx])
        for i in range(self._nstep):
            step_reward = episode['reward'][idx + i]
            reward += discount * step_reward
            discount *= episode['discount'][idx + i] * self._discount
        
        return (obs, action, reward, discount, next_obs, *meta)

    def _episode_to_transitions(self, eps_fn):
        if eps_fn in self._transition_cache:
            return self._transition_cache[eps_fn]

        episode = self._episodes[eps_fn]
        transition_count = episode_len(episode) - self._nstep + 1
        if transition_count <= 0:
            return None

        obs = episode['observation'][:transition_count]
        action = episode['action'][1:transition_count + 1]
        next_obs = episode['observation'][self._nstep:transition_count + self._nstep]

        reward = np.zeros_like(episode['reward'][1:transition_count + 1])
        discount = np.ones_like(episode['discount'][1:transition_count + 1])
        for i in range(self._nstep):
            reward += discount * episode['reward'][1 + i:transition_count + 1 + i]
            discount *= episode['discount'][1 + i:transition_count + 1 + i] * self._discount

        meta = [episode[spec.name][:transition_count] for spec in self._storage._meta_specs]
        transitions = (obs, action, reward, discount, next_obs, *meta)
        self._transition_cache[eps_fn] = transitions
        return transitions

    def _get_all_transitions(self):
        cache_key = tuple(self._episode_fns)
        if self._all_data_cache_key == cache_key and self._all_data_cache is not None:
            return self._all_data_cache

        transition_batches = []
        for eps_fn in self._episode_fns:
            transitions = self._episode_to_transitions(eps_fn)
            if transitions is not None:
                transition_batches.append(transitions)

        if not transition_batches:
            raise RuntimeError('Replay buffer is empty')

        all_transitions = tuple(
            np.concatenate([batch[field_idx] for batch in transition_batches], axis=0)
            for field_idx in range(len(transition_batches[0]))
        )
        self._all_data_cache = all_transitions
        self._all_data_cache_key = cache_key
        return all_transitions

    def _get_first_stored_transition(self):
        for eps_fn in self._episode_fns:
            transitions = self._episode_to_transitions(eps_fn)
            if transitions is None:
                continue
            return tuple(field[:1] for field in transitions)
        raise RuntimeError('Replay buffer is empty')

    def get_all_data(self, last_n=None, first=False):
        try:
            self._try_fetch(force=True)
        except:
            traceback.print_exc()

        if last_n is not None and last_n <= 0:
            raise ValueError('last_n must be positive when provided')

        if last_n is None:
            return self._get_all_transitions()

        if first:
            first_transition = self._get_first_stored_transition()
            if last_n == 1:
                return first_transition

        transition_batches = []
        remaining = last_n - 1 if first else last_n
        for eps_fn in reversed(self._episode_fns):
            if remaining == 0:
                break

            transitions = self._episode_to_transitions(eps_fn)
            if transitions is None:
                continue

            batch_size = transitions[0].shape[0]
            if batch_size > remaining:
                transitions = tuple(field[-remaining:] for field in transitions)
                batch_size = remaining

            transition_batches.append(transitions)
            remaining -= batch_size

        if not transition_batches:
            if first:
                return first_transition
            raise RuntimeError('Replay buffer is empty')

        transition_batches.reverse()
        all_transitions = tuple(
            np.concatenate([batch[field_idx] for batch in transition_batches], axis=0)
            for field_idx in range(len(transition_batches[0]))
        )

        if not first:
            return all_transitions

        return tuple(
            np.concatenate([first_transition[field_idx], all_transitions[field_idx]], axis=0)
            for field_idx in range(len(first_transition))
        )

    def __iter__(self):
        while True:
            if self._first_transition:
                # First element: first transition of a random episode
                yield self._sample(first=True)
                # Remaining elements: normal random sampling
                for _ in range(self._batch_size - 1):
                    yield self._sample()
            else:
                yield self._sample()


def _worker_init_fn(worker_id):
    seed = int(np.random.get_state()[1][0] + worker_id)
    np.random.seed(seed)
    random.seed(seed)


def make_replay_loader(storage, max_size, batch_size, num_workers,
                       save_snapshot, nstep, discount, first_transition=False):
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = ReplayBuffer(storage,
                            max_size_per_worker,
                            num_workers,
                            nstep,
                            discount,
                            fetch_every=1000,
                            save_snapshot=save_snapshot,
                            first_transition=first_transition,
                            batch_size=batch_size if first_transition else None)

    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         pin_memory=True,
                                         worker_init_fn=_worker_init_fn)
    return loader

def _relable_mujoco_episode(env, episode):
    rewards = []
    reward_spec = env.reward_spec()
    states = episode['physics']
    for i in range(states.shape[0]):
        with env.physics.reset_context():
            env.physics.set_state(states[i])
        reward = env.task.get_reward(env.physics)
        reward = np.full(reward_spec.shape, reward, reward_spec.dtype)
        rewards.append(reward)
    episode['reward'] = np.array(rewards, dtype=reward_spec.dtype)
    return episode


def _relable_from_observation_episode(env, episode):
    rewards = []
    reward_spec = env.reward_spec()
    for observation in episode['observation']:
        reward = env.compute_reward_from_observation(observation)
        reward = np.full(reward_spec.shape, reward, reward_spec.dtype)
        rewards.append(reward)
    episode['reward'] = np.array(rewards, dtype=reward_spec.dtype)
    return episode


def relable_episode(env, episode):
    if 'physics' in episode and hasattr(env, 'physics') and hasattr(env, 'task'):
        return _relable_mujoco_episode(env, episode)
    if hasattr(env, 'compute_reward_from_observation') and 'observation' in episode:
        return _relable_from_observation_episode(env, episode)
    raise NotImplementedError(
        f'Relabelling is not supported for environment type {type(env.unwrapped).__name__}'
    )


class OfflineReplayBuffer(IterableDataset):
    def __init__(self, env, replay_dir, max_size, num_workers, discount, relable=True):
        self._env = env
        self._replay_dir = replay_dir
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._discount = discount
        self._loaded = False
        self._relable = relable

    def _load(self, relable=True):
        if relable:
            print(f'Relabeling offline data for {type(self._env.unwrapped).__name__}...')
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._replay_dir.glob('*.npz'))
        print(f'Found {len(eps_fns)} episodes in replay buffer directory.')
        for eps_fn in eps_fns:
            if self._size > self._max_size:
                break
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split('_')[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            episode = load_episode(eps_fn)
            if relable:
                episode = self._relable_reward(episode)
            self._episode_fns.append(eps_fn)
            self._episodes[eps_fn] = episode
            self._size += episode_len(episode)

    def _sample_episode(self):
        if not self._loaded:
            self._load(self._relable)
            self._loaded = True
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _relable_reward(self, episode):
        return relable_episode(self._env, episode)

    def _sample(self):
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode)) + 1
        obs = episode['observation'][idx - 1]
        action = episode['action'][idx]
        next_obs = episode['observation'][idx]
        reward = episode['reward'][idx]
        discount = episode['discount'][idx] * self._discount
        return (obs, action, reward, discount, next_obs)

    def __iter__(self):
        while True:
            yield self._sample()



def make_offline_replay_loader(env, replay_dir, max_size, batch_size, num_workers,
                       discount, relable=True):
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = OfflineReplayBuffer(env, replay_dir, max_size_per_worker,
                                   num_workers, discount, relable)

    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         pin_memory=True,
                                         worker_init_fn=_worker_init_fn)
    return loader
