import numpy as np


class Discretizer:
    def __init__(self, features_ranges, bins_sizes, lambda_transform=None):
        assert len(features_ranges) == len(bins_sizes)

        self.num_features = len(features_ranges)
        self.feature_ranges = features_ranges
        self.bins_sizes = bins_sizes

        self.bins = [np.linspace(features_ranges[i][0], features_ranges[i][1], bins_sizes[i]+1)[1:-1] for i in range(self.num_features)]

        self.lambda_transform = lambda_transform


    def discretize(self, features):
        if self.lambda_transform is None:
            return tuple(np.digitize(x=features[i], bins=self.bins[i]) for i in range(len(features)))
        else:
            features = self.lambda_transform(features)
            return tuple(np.digitize(x=features[i], bins=self.bins[i]) for i in range(len(features)))


    def get_empty_mat(self):
        return np.zeros(self.bins_sizes)


class IdentityDiscretizer:
    """
    Discretizer for environments with discrete state spaces.
    Maps state indices directly to grid coordinates.
    """
    def __init__(self, bins_sizes, idx_to_state):
        self.bins_sizes = bins_sizes
        self.idx_to_state = idx_to_state
        self.num_features = len(bins_sizes)

    def discretize(self, state):
        """
        Convert state index to grid coordinates.

        Args:
            state: int (state index) or tuple (coordinates)
        """
        if isinstance(state, (int, np.integer)):
            # State is an index, convert to coordinates
            coords = self.idx_to_state[state]
            return coords
        else:
            # State is already coordinates
            return tuple(state)

    def get_empty_mat(self):
        return np.zeros(self.bins_sizes)