import sys
from pathlib import Path
import numpy as np

def load_episode(fn):
    with fn.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode

def count_rewards_in_episodes(episodes):
    reward_counts = {0: 0, -1: 0, 1: 0}
    total = 0
    for episode in episodes:
        rewards = episode['reward'].flatten()
        for r in rewards:
            if r == 0:
                reward_counts[0] += 1
            elif r == -1:
                reward_counts[-1] += 1
            elif r == 1:
                reward_counts[1] += 1
            total += 1
    return reward_counts, total

def main(replay_dir, chunk_size):
    replay_dir = Path(replay_dir)
    npz_files = sorted(replay_dir.glob('*.npz'))  # ordinati per nome
    if not npz_files:
        print(f"Nessun file .npz trovato in {replay_dir}")
        return

    chunk_size = int(chunk_size)
    n_chunks = (len(npz_files) + chunk_size - 1) // chunk_size

    total_counts = {0: 0, -1: 0, 1: 0}
    total_transitions = 0

    for i in range(n_chunks):
        chunk_files = npz_files[i*chunk_size:(i+1)*chunk_size]
        episodes = [load_episode(fn) for fn in chunk_files]
        reward_counts, transitions = count_rewards_in_episodes(episodes)
        print(f"Chunk {i}: episodi {i*chunk_size}-{min((i+1)*chunk_size-1, len(npz_files)-1)}")
        print(f"  Transizioni: {transitions}")
        print(f"  Reward 0: {reward_counts[0]}")
        print(f"  Reward -1: {reward_counts[-1]}")
        print(f"  Reward +1: {reward_counts[1]}")
        total_transitions += transitions
        for k in total_counts:
            total_counts[k] += reward_counts[k]

    print("\nTotale su tutto il dataset:")
    print(f"Transizioni totali: {total_transitions}")
    print(f"Transizioni con reward 0: {total_counts[0]}")
    print(f"Transizioni con reward -1: {total_counts[-1]}")
    print(f"Transizioni con reward +1: {total_counts[1]}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Utilizzo: python count_rewards.py <replay_buffer_dir> <chunk_size>")
    else:
        main(sys.argv[1], sys.argv[2])