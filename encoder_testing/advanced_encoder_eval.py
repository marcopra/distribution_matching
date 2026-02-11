"""
Script per valutazioni aggiuntive dell'encoder.
Include: alignment, uniformity, linear probing, nearest neighbors.
"""

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
import sys

# Aggiungi parent directory al path per importare gym_env e test_encoder_embeddings_pong
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from collections import defaultdict
import argparse
from tqdm import tqdm

# Import dal test principale
from test_encoder_embeddings_pong import EncoderTester


class AdvancedEncoderEvaluator:
    """Valutazioni avanzate dell'encoder"""
    
    def __init__(self, tester):
        self.tester = tester
        self.device = tester.device
        
    def compute_alignment(self, x, y, alpha=2):
        """
        Alignment metric da Wang & Isola (2020).
        Misura quanto sono vicini gli embedding positivi.
        Lower is better.
        """
        return (x - y).norm(dim=1).pow(alpha).mean()
    
    def compute_uniformity(self, x, t=2):
        """
        Uniformity metric da Wang & Isola (2020).
        Misura quanto sono distribuiti uniformemente gli embeddings.
        Lower is better (più uniforme).
        """
        sq_pdist = torch.pdist(x, p=2).pow(2)
        return sq_pdist.mul(-t).exp().mean().log()
    
    def evaluate_alignment_uniformity(self, episodes_data, save_dir, num_samples=1000):
        """
        Valuta alignment e uniformity per φ(s') e W·ψ(s,a).
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        
        print("\n" + "="*60)
        print("ALIGNMENT & UNIFORMITY ANALYSIS")
        print("="*60)
        
        # Raccogli coppie (s, a, s')
        all_obs = []
        all_next_obs = []
        all_actions = []
        
        for episode_name, data in episodes_data.items():
            observations = data['observations']
            actions = data['actions']
            
            for i in range(len(observations) - 1):
                all_obs.append(observations[i])
                all_actions.append(actions[i])
                all_next_obs.append(observations[i+1])
        
        # Subsample
        if len(all_obs) > num_samples:
            indices = np.random.choice(len(all_obs), num_samples, replace=False)
            all_obs = [all_obs[i] for i in indices]
            all_actions = [all_actions[i] for i in indices]
            all_next_obs = [all_next_obs[i] for i in indices]
        
        all_obs = np.array(all_obs)
        all_actions = np.array(all_actions)
        all_next_obs = np.array(all_next_obs)
        
        print(f"Numero campioni: {len(all_obs)}")
        
        # Codifica
        print("Encoding φ(s')...")
        phi_next = self.tester.encode_observations(all_next_obs)
        
        print("Encoding W·ψ(s,a)...")
        psi_sa = self.tester.encode_state_action(all_obs, all_actions)
        
        if psi_sa is None:
            print("⚠️  project_sa non disponibile, skip")
            return
        
        # Proietta ψ(s,a)
        with torch.no_grad():
            psi_sa_tensor = psi_sa.to(self.device)
            projected_psi = self.tester.agent.project_sa(psi_sa_tensor)
            projected_psi = projected_psi.cpu()
        
        # Normalizza
        phi_next_norm = F.normalize(phi_next, p=2, dim=1)
        projected_psi_norm = F.normalize(projected_psi, p=2, dim=1)
        
        # Calcola metriche
        print("\nCalcolo metriche...")
        
        alignment = self.compute_alignment(phi_next_norm, projected_psi_norm)
        uniformity_phi = self.compute_uniformity(phi_next_norm)
        uniformity_psi = self.compute_uniformity(projected_psi_norm)
        
        print("\n" + "-"*60)
        print("RISULTATI:")
        print("-"*60)
        print(f"Alignment (φ(s') vs W·ψ(s,a)):  {alignment:.6f}  (lower is better)")
        print(f"Uniformity φ(s'):               {uniformity_phi:.6f}  (lower is better)")
        print(f"Uniformity W·ψ(s,a):            {uniformity_psi:.6f}  (lower is better)")
        print("-"*60)
        
        # Interpretazione
        print("\nINTERPRETAZIONE:")
        if alignment < 0.5:
            print("  ✓ Alignment OTTIMO: φ(s') e W·ψ(s,a) molto vicini")
        elif alignment < 1.0:
            print("  ○ Alignment BUONO: φ(s') e W·ψ(s,a) abbastanza vicini")
        else:
            print("  ✗ Alignment SCARSO: φ(s') e W·ψ(s,a) lontani")
        
        if uniformity_phi < -2.0 and uniformity_psi < -2.0:
            print("  ✓ Uniformity OTTIMA: embeddings ben distribuiti")
        elif uniformity_phi < -1.0 and uniformity_psi < -1.0:
            print("  ○ Uniformity BUONA: embeddings distribuiti")
        else:
            print("  ✗ Uniformity SCARSA: possibile collapsing")
        
        # Salva risultati
        results = {
            'alignment': alignment.item(),
            'uniformity_phi': uniformity_phi.item(),
            'uniformity_psi': uniformity_psi.item()
        }
        
        np.save(save_dir / 'alignment_uniformity.npy', results)
        print(f"\n✓ Risultati salvati in: {save_dir / 'alignment_uniformity.npy'}")
        
        return results
    
    def linear_probing(self, episodes_data, save_dir):
        """
        Linear probing: addestra classificatore lineare per predire tipo episodio.
        Se embeddings sono buoni, classificatore lineare dovrebbe funzionare bene.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        
        print("\n" + "="*60)
        print("LINEAR PROBING")
        print("="*60)
        
        # Prepara dati
        all_embeddings = []
        all_labels = []
        label_to_idx = {}
        idx = 0
        
        for episode_name, data in tqdm(episodes_data.items(), desc="Encoding episodes"):
            observations = data['observations']
            embeddings = self.tester.encode_observations(observations)
            
            # Etichetta = tipo episodio (prima parte del nome)
            ep_type = '_'.join(episode_name.split('_')[:-1])
            
            if ep_type not in label_to_idx:
                label_to_idx[ep_type] = idx
                idx += 1
            
            label = label_to_idx[ep_type]
            
            all_embeddings.append(embeddings)
            all_labels.extend([label] * len(embeddings))
        
        # Concatena
        all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
        all_labels = np.array(all_labels)
        
        print(f"Totale campioni: {len(all_embeddings)}")
        print(f"Numero classi: {len(label_to_idx)}")
        print(f"Classi: {label_to_idx}")
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            all_embeddings, all_labels, test_size=0.3, random_state=42, stratify=all_labels
        )
        
        print(f"\nTrain samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        # Addestra classificatore
        print("\nTraining logistic regression...")
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        
        # Valuta
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        
        print("\n" + "-"*60)
        print("RISULTATI:")
        print("-"*60)
        print(f"Train Accuracy: {train_acc:.2%}")
        print(f"Test Accuracy:  {test_acc:.2%}")
        print("-"*60)
        
        # Baseline (random guessing)
        baseline = 1.0 / len(label_to_idx)
        print(f"\nBaseline (random): {baseline:.2%}")
        
        if test_acc > baseline * 2:
            print("✓ Classificatore MOLTO MIGLIORE del random")
        elif test_acc > baseline * 1.5:
            print("○ Classificatore MIGLIORE del random")
        else:
            print("✗ Classificatore SCARSO (embeddings poco informativi)")
        
        # Classification report
        y_pred = clf.predict(X_test)
        idx_to_label = {v: k for k, v in label_to_idx.items()}
        target_names = [idx_to_label[i] for i in range(len(label_to_idx))]
        
        print("\n" + "-"*60)
        print("Classification Report:")
        print("-"*60)
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=target_names, yticklabels=target_names, ax=ax)
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        ax.set_title('Confusion Matrix - Linear Probe', fontsize=14, fontweight='bold')
        
        save_path = save_dir / 'linear_probe_confusion_matrix.png'
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Confusion matrix salvata: {save_path}")
        
        return {'train_acc': train_acc, 'test_acc': test_acc}
    
    def intrinsic_dimension_analysis(self, episodes_data, save_dir):
        """
        Analizza la dimensione intrinseca degli embeddings usando PCA.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        
        print("\n" + "="*60)
        print("INTRINSIC DIMENSION ANALYSIS")
        print("="*60)
        
        # Raccogli tutti gli embeddings
        all_embeddings = []
        
        for episode_name, data in tqdm(episodes_data.items(), desc="Encoding episodes"):
            observations = data['observations']
            embeddings = self.tester.encode_observations(observations)
            all_embeddings.append(embeddings)
        
        all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
        
        print(f"Totale embeddings: {len(all_embeddings)}")
        print(f"Embedding dimension: {all_embeddings.shape[1]}")
        
        # PCA
        print("\nCalcolo PCA...")
        pca = PCA()
        pca.fit(all_embeddings)
        
        # Varianza spiegata
        explained_var = pca.explained_variance_ratio_
        cumsum_var = np.cumsum(explained_var)
        
        # Trova dimensione intrinseca (95% varianza)
        intrinsic_dim_95 = np.argmax(cumsum_var >= 0.95) + 1
        intrinsic_dim_99 = np.argmax(cumsum_var >= 0.99) + 1
        
        print(f"\nDimensione per 95% varianza: {intrinsic_dim_95}")
        print(f"Dimensione per 99% varianza: {intrinsic_dim_99}")
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Varianza spiegata per componente
        axes[0].bar(range(1, min(51, len(explained_var)+1)), explained_var[:50], alpha=0.7)
        axes[0].set_xlabel('Principal Component', fontsize=12)
        axes[0].set_ylabel('Explained Variance Ratio', fontsize=12)
        axes[0].set_title('Variance Explained per Component', fontsize=12, fontweight='bold')
        axes[0].grid(alpha=0.3)
        
        # Plot 2: Varianza cumulativa
        axes[1].plot(range(1, len(cumsum_var)+1), cumsum_var, linewidth=2)
        axes[1].axhline(0.95, color='r', linestyle='--', label='95% variance')
        axes[1].axhline(0.99, color='orange', linestyle='--', label='99% variance')
        axes[1].axvline(intrinsic_dim_95, color='r', linestyle=':', alpha=0.5)
        axes[1].axvline(intrinsic_dim_99, color='orange', linestyle=':', alpha=0.5)
        axes[1].set_xlabel('Number of Components', fontsize=12)
        axes[1].set_ylabel('Cumulative Explained Variance', fontsize=12)
        axes[1].set_title('Cumulative Variance Explained', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        save_path = save_dir / 'intrinsic_dimension.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Plot salvato: {save_path}")
        
        # Interpretazione
        print("\nINTERPRETAZIONE:")
        ratio = intrinsic_dim_95 / all_embeddings.shape[1]
        if ratio < 0.3:
            print(f"  ✓ Dimensione intrinseca BASSA ({intrinsic_dim_95}/{all_embeddings.shape[1]} = {ratio:.1%})")
            print("    Embeddings compatti e informativi")
        elif ratio < 0.7:
            print(f"  ○ Dimensione intrinseca MEDIA ({intrinsic_dim_95}/{all_embeddings.shape[1]} = {ratio:.1%})")
            print("    Embeddings ragionevolmente compatti")
        else:
            print(f"  ⚠ Dimensione intrinseca ALTA ({intrinsic_dim_95}/{all_embeddings.shape[1]} = {ratio:.1%})")
            print("    Possibile ridondanza negli embeddings")
        
        return {
            'intrinsic_dim_95': intrinsic_dim_95,
            'intrinsic_dim_99': intrinsic_dim_99,
            'total_dim': all_embeddings.shape[1]
        }


def main():
    parser = argparse.ArgumentParser(description='Valutazioni avanzate encoder')
    parser.add_argument('--snapshot', type=str, required=True,
                       help='Path allo snapshot.pt')
    parser.add_argument('--episodes_dir', type=str, default='./salient_episodes_pong',
                       help='Directory episodi')
    parser.add_argument('--save_dir', type=str, default='./encoder_advanced_results',
                       help='Directory risultati')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Numero campioni per alignment/uniformity')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("VALUTAZIONI AVANZATE ENCODER")
    print("=" * 60)
    print(f"Snapshot: {args.snapshot}")
    print(f"Episodes dir: {args.episodes_dir}")
    print(f"Save dir: {args.save_dir}")
    print("=" * 60)
    
    # Carica encoder
    tester = EncoderTester(args.snapshot, device=args.device)
    
    # Carica episodi
    episodes_data = tester.load_episodes(args.episodes_dir)
    
    if len(episodes_data) == 0:
        print("⚠️  Nessun episodio trovato!")
        return
    
    # Crea evaluator
    evaluator = AdvancedEncoderEvaluator(tester)
    
    # Directory risultati
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Esegui valutazioni
    print("\n" + "=" * 60)
    print("INIZIO VALUTAZIONI")
    print("=" * 60)
    
    # 1. Alignment & Uniformity
    alignment_results = evaluator.evaluate_alignment_uniformity(
        episodes_data, save_dir, num_samples=args.num_samples
    )
    
    # 2. Linear Probing
    linear_probe_results = evaluator.linear_probing(episodes_data, save_dir)
    
    # 3. Intrinsic Dimension
    intrinsic_dim_results = evaluator.intrinsic_dimension_analysis(episodes_data, save_dir)
    
    # Salva tutti i risultati
    all_results = {
        'alignment_uniformity': alignment_results if alignment_results else None,
        'linear_probe': linear_probe_results,
        'intrinsic_dimension': intrinsic_dim_results
    }
    
    np.save(save_dir / 'all_advanced_results.npy', all_results)
    
    print("\n" + "=" * 60)
    print("VALUTAZIONI COMPLETATE")
    print("=" * 60)
    print(f"Risultati salvati in: {save_dir}")
    print("\nFile generati:")
    print("  - alignment_uniformity.npy")
    print("  - linear_probe_confusion_matrix.png")
    print("  - intrinsic_dimension.png")
    print("  - all_advanced_results.npy")


if __name__ == '__main__':
    main()
