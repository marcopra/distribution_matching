from collections import OrderedDict

import numpy as np


class RandomAgent:
    """Stateless agent that samples every action uniformly at random."""

    def __init__(self, action_shape, **kwargs):
        del kwargs
        if len(action_shape) != 1 or int(action_shape[0]) <= 0:
            raise ValueError(
                "RandomAgent expects a discrete action shape containing the number "
                f"of actions, got {action_shape}."
            )
        self.num_actions = int(action_shape[0])
        self.training = False
        self.requires_replay = False

        # pretrain_parallel inspects these schedule attributes.
        self.update_every_steps = 1
        self.update_actor_every_steps = 1
        self.num_expl_steps = 0
        self.T_init_steps = 0

    def train(self, training=True):
        self.training = training

    def get_meta_specs(self):
        return tuple()

    def init_meta(self):
        return OrderedDict()

    def update_meta(self, meta, global_step, time_step, finetune=False):
        del global_step, time_step, finetune
        return meta

    def act(self, obs, meta, step, eval_mode):
        del obs, meta, step, eval_mode
        return np.random.randint(self.num_actions)

    def act_parallel(self, observations, metas, steps, eval_mode):
        del metas, steps, eval_mode
        return np.random.randint(
            self.num_actions, size=len(observations), dtype=np.int64
        )

    def update(self, replay_iter, step):
        del replay_iter, step
        return {}
