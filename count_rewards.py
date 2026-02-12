import sys
from pathlib import Path
import numpy as np

def load_episode(fn):
    with fn.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode

def main(replay_dir):
    replay_dir = Path(replay_dir)
    reward_counts = {0: 0, -1: 0, 1: 0}
    total = 0

    npz_files = list(replay_dir.glob('*.npz'))
    if not npz_files:
        print(f"Nessun file .npz trovato in {replay_dir}")
        return

    for fn in npz_files:
        episode = load_episode(fn)
        rewards = episode['reward'].flatten()
        for r in rewards:
            if r == 0:
                reward_counts[0] += 1
            elif r == -1:
                reward_counts[-1] += 1
            elif r == 1:
                reward_counts[1] += 1
            total += 1

    print(f"Transizioni totali: {total}")
    print(f"Transizioni con reward 0: {reward_counts[0]}")
    print(f"Transizioni con reward -1: {reward_counts[-1]}")
    print(f"Transizioni con reward +1: {reward_counts[1]}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Utilizzo: python count_rewards.py <replay_buffer_dir>")
    else:
        main(sys.argv[1])