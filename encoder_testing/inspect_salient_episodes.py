"""
Script di utilità per ispezionare episodi salienti salvati.
Mostra statistiche e permette di verificare i dati.
"""

import numpy as np
from pathlib import Path
import argparse
from collections import defaultdict


def inspect_episodes(episodes_dir):
    """Ispeziona e mostra statistiche degli episodi salvati"""
    
    episodes_dir = Path(episodes_dir)
    
    if not episodes_dir.exists():
        print(f"❌ Directory non trovata: {episodes_dir}")
        return
    
    npz_files = sorted(episodes_dir.glob('*.npz'))
    mp4_files = sorted(episodes_dir.glob('*.mp4'))
    
    print("=" * 70)
    print("ISPEZIONE EPISODI SALIENTI - PONG")
    print("=" * 70)
    print(f"Directory: {episodes_dir}")
    print(f"File .npz trovati: {len(npz_files)}")
    print(f"File .mp4 trovati: {len(mp4_files)}")
    print("=" * 70)
    
    if len(npz_files) == 0:
        print("\n⚠️  Nessun episodio trovato!")
        return
    
    # Raggruppa per tipo
    episode_types = defaultdict(list)
    
    print("\nDETTAGLI EPISODI:")
    print("-" * 70)
    print(f"{'Nome Episodio':<40} {'Steps':>8} {'Reward':>10} {'Video':>6}")
    print("-" * 70)
    
    total_steps = 0
    total_reward = 0
    
    for npz_file in npz_files:
        episode_name = npz_file.stem
        
        # Carica dati
        data = np.load(npz_file)
        
        observations = data['observations']
        rewards = data['rewards']
        
        num_steps = len(observations)
        total_rew = rewards.sum()
        
        # Verifica se esiste video
        video_file = episodes_dir / f'{episode_name}.mp4'
        has_video = "✓" if video_file.exists() else "✗"
        
        # Tipo episodio
        ep_type = '_'.join(episode_name.split('_')[:-1])
        episode_types[ep_type].append(episode_name)
        
        print(f"{episode_name:<40} {num_steps:>8} {total_rew:>10.1f} {has_video:>6}")
        
        total_steps += num_steps
        total_reward += total_rew
    
    print("-" * 70)
    print(f"{'TOTALE':<40} {total_steps:>8} {total_reward:>10.1f}")
    print("-" * 70)
    
    # Statistiche per tipo
    print("\n" + "=" * 70)
    print("STATISTICHE PER TIPO")
    print("=" * 70)
    print(f"{'Tipo Episodio':<40} {'Numero Episodi':>20}")
    print("-" * 70)
    
    for ep_type, episodes in sorted(episode_types.items()):
        print(f"{ep_type:<40} {len(episodes):>20}")
    
    print("=" * 70)
    
    # Verifica shape delle osservazioni
    print("\n" + "=" * 70)
    print("INFORMAZIONI OSSERVAZIONI")
    print("=" * 70)
    
    first_npz = npz_files[0]
    data = np.load(first_npz)
    
    print(f"File campione: {first_npz.name}")
    print(f"Observation shape: {data['observations'].shape}")
    print(f"Observation dtype: {data['observations'].dtype}")
    print(f"Actions shape: {data['actions'].shape}")
    print(f"Actions dtype: {data['actions'].dtype}")
    print(f"Rewards shape: {data['rewards'].shape}")
    print(f"Frames shape: {data['frames'].shape}")
    print("=" * 70)
    
    # Verifica consistenza
    print("\nVERIFICA CONSISTENZA:")
    inconsistencies = 0
    
    for npz_file in npz_files:
        data = np.load(npz_file)
        
        obs_len = len(data['observations'])
        act_len = len(data['actions'])
        rew_len = len(data['rewards'])
        
        if not (obs_len == act_len == rew_len):
            print(f"  ⚠️  {npz_file.name}: Lunghezze diverse (obs={obs_len}, act={act_len}, rew={rew_len})")
            inconsistencies += 1
    
    if inconsistencies == 0:
        print("  ✓ Tutti gli episodi sono consistenti")
    else:
        print(f"  ⚠️  {inconsistencies} episodi con inconsistenze")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Ispeziona episodi salienti salvati')
    parser.add_argument('--episodes_dir', type=str, default='./salient_episodes_pong',
                       help='Directory con episodi salvati')
    
    args = parser.parse_args()
    
    inspect_episodes(args.episodes_dir)


if __name__ == '__main__':
    main()
