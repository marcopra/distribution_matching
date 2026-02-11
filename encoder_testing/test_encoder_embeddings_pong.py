"""
Script per testare l'encoder dell'algoritmo con episodi salienti da Pong.
Visualizza embeddings con t-SNE e analizza la qualità del contrastive learning.
"""

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
import sys

# Aggiungi parent directory al path per importare gym_env
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import Patch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
from collections import defaultdict
import argparse
from tqdm import tqdm
import gym_env


class EncoderTester:
    """Testa e visualizza embeddings dell'encoder"""
    
    def __init__(self, snapshot_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.snapshot_path = Path(snapshot_path)
        self.agent = None
        self.encoder = None
        
        # Carica snapshot
        self._load_snapshot()
        
    def _load_snapshot(self):
        """Carica snapshot dell'agente"""
        print(f"Caricamento snapshot da: {self.snapshot_path}")
        
        if not self.snapshot_path.exists():
            raise FileNotFoundError(f"Snapshot non trovato: {self.snapshot_path}")
        
        # Carica snapshot
        snapshot = torch.load(self.snapshot_path, map_location=self.device, weights_only=False)
        self.agent = snapshot['agent']
        
        # Estrai encoder
        if hasattr(self.agent, 'encoder'):
            self.encoder = self.agent.encoder
            print(f"✓ Encoder caricato: {type(self.encoder).__name__}")
        else:
            raise ValueError("L'agente non ha un encoder!")
        
        # Metti in eval mode
        self.encoder.eval()
        
        # Verifica altre componenti utili
        self.has_projector = hasattr(self.encoder, 'encode_and_project')
        self.has_project_sa = hasattr(self.agent, 'project_sa')
        
        print(f"  - Encoder has projector: {self.has_projector}")
        print(f"  - Agent has project_sa: {self.has_project_sa}")
        
    def load_episodes(self, episodes_dir):
        """
        Carica episodi salvati da directory.
        
        Returns:
            episodes_data: Dict[episode_name] -> dict con observations, actions, etc.
        """
        episodes_dir = Path(episodes_dir)
        if not episodes_dir.exists():
            raise FileNotFoundError(f"Directory episodi non trovata: {episodes_dir}")
        
        episodes_data = {}
        npz_files = sorted(episodes_dir.glob('*.npz'))
        
        print(f"\nCaricamento episodi da: {episodes_dir}")
        print(f"Trovati {len(npz_files)} file .npz")
        
        for npz_file in npz_files:
            episode_name = npz_file.stem
            data = np.load(npz_file)
            
            episodes_data[episode_name] = {
                'observations': data['observations'],
                'actions': data['actions'],
                'rewards': data['rewards'],
            }
            
            print(f"  ✓ {episode_name}: {len(data['observations'])} steps")
        
        return episodes_data
    
    def encode_observations(self, observations, batch_size=128):
        """
        Passa osservazioni attraverso l'encoder.
        
        Args:
            observations: Array di osservazioni [N, ...]
            batch_size: Batch size per encoding
            
        Returns:
            embeddings: Tensor [N, feature_dim]
        """
        all_embeddings = []
        
        # Converti in tensor e processa in batch
        observations = torch.FloatTensor(observations).to(self.device)
        
        with torch.no_grad():
            for i in range(0, len(observations), batch_size):
                batch = observations[i:i+batch_size]
                
                # Encoding
                if self.has_projector:
                    embeddings = self.encoder.encode_and_project(batch)
                else:
                    embeddings = self.encoder(batch)
                
                all_embeddings.append(embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0)
    
    def encode_state_action(self, observations, actions, batch_size=128):
        """
        Codifica coppie stato-azione usando project_sa.
        
        Args:
            observations: Array di osservazioni [N, ...]
            actions: Array di azioni [N]
            batch_size: Batch size
            
        Returns:
            psi_embeddings: Tensor [N, feature_dim]
        """
        if not self.has_project_sa:
            print("⚠️  Agent non ha project_sa, ritorno None")
            return None
        
        all_psi = []
        
        observations = torch.FloatTensor(observations).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        
        with torch.no_grad():
            for i in range(0, len(observations), batch_size):
                obs_batch = observations[i:i+batch_size]
                action_batch = actions[i:i+batch_size]
                
                # Codifica osservazione
                if self.has_projector:
                    encoded_obs = self.encoder.encode_and_project(obs_batch)
                else:
                    encoded_obs = self.encoder(obs_batch)
                
                # Codifica stato-azione
                psi = self.agent._encode_state_action(encoded_obs, action_batch)
                all_psi.append(psi.cpu())
        
        return torch.cat(all_psi, dim=0)
    
    def compute_tsne(self, embeddings, perplexity=10, n_iter=1000):
        """Calcola t-SNE per embeddings"""
        embeddings_np = embeddings.numpy() if torch.is_tensor(embeddings) else embeddings
        
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings_np)
        
        return embeddings_2d
    
    def plot_tsne_per_trajectory(self, episodes_data, save_dir):
        """
        Plot t-SNE separato per ogni traiettoria.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        
        print("\n" + "="*60)
        print("PLOT t-SNE PER TRAIETTORIA")
        print("="*60)
        
        for episode_name, data in tqdm(episodes_data.items(), desc="Processing episodes"):
            observations = data['observations']
            
            # Codifica osservazioni
            embeddings = self.encode_observations(observations)
            
            # Calcola t-SNE
            embeddings_2d = self.compute_tsne(embeddings)
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Colora per timestep (gradiente temporale)
            timesteps = np.arange(len(embeddings_2d))
            scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                               c=timesteps, cmap='viridis', s=50, alpha=0.6)
            
            # Aggiungi frecce per mostrare direzione temporale
            step_size = max(1, len(embeddings_2d) // 20)
            for i in range(0, len(embeddings_2d) - step_size, step_size):
                ax.arrow(embeddings_2d[i, 0], embeddings_2d[i, 1],
                        embeddings_2d[i+step_size, 0] - embeddings_2d[i, 0],
                        embeddings_2d[i+step_size, 1] - embeddings_2d[i, 1],
                        head_width=0.3, head_length=0.2, fc='red', ec='red', alpha=0.3)
            
            plt.colorbar(scatter, label='Timestep')
            ax.set_xlabel('t-SNE Component 1', fontsize=12)
            ax.set_ylabel('t-SNE Component 2', fontsize=12)
            ax.set_title(f't-SNE: {episode_name}\n({len(embeddings)} states)', fontsize=14, fontweight='bold')
            ax.grid(alpha=0.3)
            
            # Salva
            save_path = save_dir / f'tsne_{episode_name}.png'
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Salvato: {save_path.name}")
        
        print(f"\n✓ Plot salvati in: {save_dir}")
    
    def plot_tsne_all_trajectories(self, episodes_data, save_dir):
        """
        Plot t-SNE con tutte le traiettorie insieme (colori diversi).
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        
        print("\n" + "="*60)
        print("PLOT t-SNE TUTTE LE TRAIETTORIE")
        print("="*60)
        
        # Raccogli tutti gli embeddings
        all_embeddings = []
        all_labels = []
        episode_names = []
        
        for episode_name, data in tqdm(episodes_data.items(), desc="Encoding episodes"):
            observations = data['observations']
            embeddings = self.encode_observations(observations)
            
            all_embeddings.append(embeddings)
            all_labels.extend([episode_name] * len(embeddings))
            episode_names.append(episode_name)
        
        # Concatena
        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        print(f"Totale stati: {len(all_embeddings)}")
        print(f"Numero traiettorie: {len(episode_names)}")
        
        # Calcola t-SNE
        print("Calcolo t-SNE...")
        embeddings_2d = self.compute_tsne(all_embeddings)
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Crea color map
        colors = plt.cm.tab20(np.linspace(0, 1, len(episode_names)))
        color_map = {name: colors[i] for i, name in enumerate(episode_names)}
        
        # Plot per traiettoria
        for episode_name in episode_names:
            mask = np.array([label == episode_name for label in all_labels])
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                      c=[color_map[episode_name]], label=episode_name,
                      s=30, alpha=0.6)
        
        ax.set_xlabel('t-SNE Component 1', fontsize=12)
        ax.set_ylabel('t-SNE Component 2', fontsize=12)
        ax.set_title(f't-SNE: Tutte le Traiettorie\n({len(all_embeddings)} stati totali)', 
                    fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(alpha=0.3)
        
        # Salva
        save_path = save_dir / 'tsne_all_trajectories.png'
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Plot salvato: {save_path}")
        
    def plot_phi_vs_projected_psi(self, episodes_data, save_dir, num_samples=1000):
        """
        Plot φ(s') vs W·ψ(s,a) per verificare se vengono mappati vicini.
        Questo testa l'obiettivo del contrastive learning.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        
        print("\n" + "="*60)
        print("PLOT φ(s') vs W·ψ(s,a)")
        print("="*60)
        
        if not self.has_project_sa:
            print("⚠️  Agent non ha project_sa, skip questo test")
            return
        
        # Raccogli campioni da tutti gli episodi
        all_obs = []
        all_next_obs = []
        all_actions = []
        
        for episode_name, data in episodes_data.items():
            observations = data['observations']
            actions = data['actions']
            
            # Crea coppie (s, a, s')
            for i in range(len(observations) - 1):
                all_obs.append(observations[i])
                all_actions.append(actions[i])
                all_next_obs.append(observations[i+1])
        
        # Subsample se troppi
        if len(all_obs) > num_samples:
            indices = np.random.choice(len(all_obs), num_samples, replace=False)
            all_obs = [all_obs[i] for i in indices]
            all_actions = [all_actions[i] for i in indices]
            all_next_obs = [all_next_obs[i] for i in indices]
        
        all_obs = np.array(all_obs)
        all_actions = np.array(all_actions)
        all_next_obs = np.array(all_next_obs)
        
        print(f"Numero campioni: {len(all_obs)}")

        max_len = min(500, len(all_obs))
        
        # Codifica φ(s')
        print("Encoding φ(s')...")
        phi_next = self.encode_observations(all_next_obs)
        
        # Codifica ψ(s,a) e proietta
        print("Encoding W·ψ(s,a)...")
        psi_sa = self.encode_state_action(all_obs, all_actions)
        
        if psi_sa is None:
            return
        
        # Proietta ψ(s,a) -> φ space usando project_sa
        with torch.no_grad():
            psi_sa_tensor = psi_sa.to(self.device)
            projected_psi = self.agent.project_sa(psi_sa_tensor)
            projected_psi = projected_psi.cpu()
        
        # Calcola similarità coseno
        phi_next_norm = F.normalize(phi_next, p=2, dim=1)
        projected_psi_norm = F.normalize(projected_psi, p=2, dim=1)
        cosine_similarities = (phi_next_norm * projected_psi_norm).sum(dim=1)
        
        # Calcola distanze L2
        l2_distances = torch.norm(phi_next - projected_psi, p=2, dim=1)
        
        print(f"Cosine similarity - Mean: {cosine_similarities.mean():.4f}, Std: {cosine_similarities.std():.4f}")
        print(f"L2 distance - Mean: {l2_distances.mean():.4f}, Std: {l2_distances.std():.4f}")
        
        # Plot 1: Histogram delle similarità
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Cosine similarity distribution
        axes[0, 0].hist(cosine_similarities.numpy(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].axvline(cosine_similarities.mean().item(), color='red', 
                          linestyle='--', linewidth=2, label=f'Mean: {cosine_similarities.mean():.3f}')
        axes[0, 0].set_xlabel('Cosine Similarity', fontsize=11)
        axes[0, 0].set_ylabel('Frequency', fontsize=11)
        axes[0, 0].set_title('Distribution of Cosine Similarities\nφ(s\') vs W·ψ(s,a)', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # L2 distance distribution
        axes[0, 1].hist(l2_distances.numpy(), bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].axvline(l2_distances.mean().item(), color='red',
                          linestyle='--', linewidth=2, label=f'Mean: {l2_distances.mean():.3f}')
        axes[0, 1].set_xlabel('L2 Distance', fontsize=11)
        axes[0, 1].set_ylabel('Frequency', fontsize=11)
        axes[0, 1].set_title('Distribution of L2 Distances\nφ(s\') vs W·ψ(s,a)', fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Plot 2: t-SNE di entrambi insieme
        print("Calcolo t-SNE per visualizzazione congiunta...")
        combined = torch.cat([phi_next[:max_len], projected_psi[:max_len]], dim=0)
        labels = ['φ(s\')'] * max_len + ['W·ψ(s,a)'] * max_len
        
        combined_2d = self.compute_tsne(combined)
        
        colors_map = {'φ(s\')': 'blue', 'W·ψ(s,a)': 'red'}
        for label in ['φ(s\')', 'W·ψ(s,a)']:
            mask = np.array([l == label for l in labels])
            axes[1, 0].scatter(combined_2d[mask, 0], combined_2d[mask, 1],
                             c=colors_map[label], label=label, s=20, alpha=0.5)
        
        axes[1, 0].set_xlabel('t-SNE Component 1', fontsize=11)
        axes[1, 0].set_ylabel('t-SNE Component 2', fontsize=11)
        axes[1, 0].set_title('t-SNE: φ(s\') vs W·ψ(s,a)', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Plot 3: Scatter plot per-dimension correlation
        # Prendiamo le prime 2 dimensioni come esempio
        axes[1, 1].scatter(phi_next[:, 0].numpy(), projected_psi[:, 0].numpy(),
                          s=10, alpha=0.3, c='purple')
        axes[1, 1].set_xlabel('φ(s\') - Dim 0', fontsize=11)
        axes[1, 1].set_ylabel('W·ψ(s,a) - Dim 0', fontsize=11)
        axes[1, 1].set_title('Correlation: Dimension 0', fontsize=12, fontweight='bold')
        axes[1, 1].grid(alpha=0.3)
        
        # Linea ideale y=x
        lims = [
            np.min([axes[1, 1].get_xlim(), axes[1, 1].get_ylim()]),
            np.max([axes[1, 1].get_xlim(), axes[1, 1].get_ylim()]),
        ]
        axes[1, 1].plot(lims, lims, 'r--', alpha=0.5, linewidth=2, label='y=x')
        axes[1, 1].legend()
        
        plt.tight_layout()
        save_path = save_dir / 'phi_vs_projected_psi.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Plot salvato: {save_path}")
        
    def analyze_embedding_quality(self, episodes_data, save_dir):
        """
        Analisi qualitativa aggiuntiva degli embeddings.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        
        print("\n" + "="*60)
        print("ANALISI QUALITÀ EMBEDDINGS")
        print("="*60)
        
        # 1. Separation between episode types
        print("\n1. Analisi separazione per tipo di episodio...")
        
        episode_types = defaultdict(list)
        for episode_name in episodes_data.keys():
            # Estrai tipo (prima parte del nome)
            ep_type = '_'.join(episode_name.split('_')[:-1])
            episode_types[ep_type].append(episode_name)
        
        # Calcola embedding medio per tipo
        type_centroids = {}
        for ep_type, ep_names in episode_types.items():
            all_embeddings = []
            for ep_name in ep_names:
                obs = episodes_data[ep_name]['observations']
                emb = self.encode_observations(obs)
                all_embeddings.append(emb)
            
            # Media
            all_embeddings = torch.cat(all_embeddings, dim=0)
            centroid = all_embeddings.mean(dim=0)
            type_centroids[ep_type] = centroid
        
        # Calcola distanze tra centroidi
        print("\nDistanze tra centroidi di tipi diversi:")
        types_list = list(type_centroids.keys())
        for i, type1 in enumerate(types_list):
            for type2 in types_list[i+1:]:
                dist = torch.norm(type_centroids[type1] - type_centroids[type2], p=2)
                print(f"  {type1:25s} <-> {type2:25s}: {dist:.4f}")
        
        # 2. Temporal consistency
        print("\n2. Analisi consistenza temporale...")
        temporal_dists = []
        
        for episode_name, data in episodes_data.items():
            obs = data['observations']
            emb = self.encode_observations(obs)
            
            # Distanza tra stati consecutivi
            for i in range(len(emb) - 1):
                dist = torch.norm(emb[i+1] - emb[i], p=2)
                temporal_dists.append(dist.item())
        
        temporal_dists = np.array(temporal_dists)
        print(f"Distanza temporale media: {temporal_dists.mean():.4f} ± {temporal_dists.std():.4f}")
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(temporal_dists, bins=50, alpha=0.7, color='cyan', edgecolor='black')
        ax.axvline(temporal_dists.mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {temporal_dists.mean():.3f}')
        ax.set_xlabel('L2 Distance Between Consecutive States', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Temporal Consistency of Embeddings', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        save_path = save_dir / 'temporal_consistency.png'
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Plot salvato: {save_path}")
        
        # 3. Embedding norm distribution
        print("\n3. Analisi distribuzione norme embeddings...")
        all_norms = []
        for episode_name, data in episodes_data.items():
            obs = data['observations']
            emb = self.encode_observations(obs)
            norms = torch.norm(emb, p=2, dim=1)
            all_norms.extend(norms.tolist())
        
        all_norms = np.array(all_norms)
        print(f"Norma embeddings - Mean: {all_norms.mean():.4f}, Std: {all_norms.std():.4f}")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(all_norms, bins=50, alpha=0.7, color='orange', edgecolor='black')
        ax.axvline(all_norms.mean(), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {all_norms.mean():.3f}')
        ax.set_xlabel('L2 Norm', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Embedding Norms', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        save_path = save_dir / 'embedding_norms.png'
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Plot salvato: {save_path}")
        
        print("\n✓ Analisi completata")


def main():
    parser = argparse.ArgumentParser(description='Testa encoder con episodi salienti da Pong')
    parser.add_argument('--snapshot', type=str, required=True,
                       help='Path allo snapshot.pt dell\'agente')
    parser.add_argument('--episodes_dir', type=str, default='./salient_episodes_pong',
                       help='Directory con episodi salvati')
    parser.add_argument('--save_dir', type=str, default='./encoder_test_results',
                       help='Directory dove salvare i plot')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda o cpu)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TEST ENCODER - EPISODI SALIENTI PONG")
    print("=" * 60)
    print(f"Snapshot: {args.snapshot}")
    print(f"Episodes dir: {args.episodes_dir}")
    print(f"Save dir: {args.save_dir}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    # Crea tester
    tester = EncoderTester(args.snapshot, device=args.device)
    
    # Carica episodi
    episodes_data = tester.load_episodes(args.episodes_dir)
    
    if len(episodes_data) == 0:
        print("⚠️  Nessun episodio trovato!")
        return
    
    # Crea directory per i risultati
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Esegui test
    print("\n" + "=" * 60)
    print("INIZIO TEST")
    print("=" * 60)
    
    # 1. t-SNE per traiettoria
    tester.plot_tsne_per_trajectory(episodes_data, save_dir / 'tsne_per_trajectory')
    
    # 2. t-SNE tutte le traiettorie
    tester.plot_tsne_all_trajectories(episodes_data, save_dir)
    
    # 3. φ(s') vs W·ψ(s,a)
    tester.plot_phi_vs_projected_psi(episodes_data, save_dir)
    
    # 4. Analisi qualità
    tester.analyze_embedding_quality(episodes_data, save_dir)
    
    print("\n" + "=" * 60)
    print("TEST COMPLETATO")
    print("=" * 60)
    print(f"Tutti i risultati salvati in: {save_dir}")


if __name__ == '__main__':
    main()
