# Quick Start Guide - Encoder Testing su Pong

> **Nota**: Tutti gli script di testing sono nella cartella `encoder_testing/`. 
> Esegui i comandi dalla directory principale del progetto usando `encoder_testing/script.py` 
> oppure entra nella cartella: `cd encoder_testing` e poi esegui gli script.

## Setup Rapido

### 1. Training (se non hai già uno snapshot)

```bash
# Train con configurazione Pong
python pretrain.py --config-name=configs/pretrain_pong_v2

# Oppure usa una configurazione esistente
python pretrain.py task_name=PongNoFrameskip-v4 agent=dist_matching_embedding_augmented_v2
```

### 2. Testing Encoder - Workflow Automatico

**Opzione A: Workflow completo (sampling + testing)**

```bash
# Dalla directory principale del progetto:
encoder_testing/run_encoder_test.sh exp_local/2026.02.11/123456/snapshots/snapshot_100000.pt

# Oppure:
cd encoder_testing
./run_encoder_test.sh ../exp_local/2026.02.11/123456/snapshots/snapshot_100000.pt
```

**Opzione B: Solo testing (usa episodi già campionati)**

```bash
cd encoder_testing
./run_encoder_test.sh ../exp_local/2026.02.11/123456/snapshots/snapshot_100000.pt --skip-sampling
```

**Opzione C: Custom directories**

```bash
cd encoder_testing
./run_encoder_test.sh ../exp_local/2026.02.11/123456/snapshots/snapshot_100000.pt \
    --episodes-dir ../my_episodes \
    --results-dir ../my_results \
    --num-episodes 10
```

### 3. Testing Encoder - Manuale (Step-by-Step)

**Step 1: Campiona episodi salienti**

```bash
python sample_salient_episodes_pong.py \
    --save_dir ./salient_episodes \
    --num_episodes 5 \
    --seed 42
```

**Step 2: Ispeziona episodi (opzionale)**

```bash
python inspect_salient_episodes.py --episodes_dir ./salient_episodes
```

**Step 3: Testa encoder**

```bash
python test_encoder_embeddings_pong.py \
    --snapshot exp_local/2026.02.11/123456/snapshots/snapshot_100000.pt \
    --episodes_dir ./salient_episodes \
    --save_dir ./results
```

**Step 4: Valutazioni avanzate (opzionale)**

```bash
python advanced_encoder_eval.py \
    --snapshot exp_local/2026.02.11/123456/snapshots/snapshot_100000.pt \
    --episodes_dir ./salient_episodes \
    --save_dir ./advanced_results
```

## Esempi Specifici

### Testare encoder a diversi checkpoint

```bash
# Test a 100k frames
./run_encoder_test.sh exp_local/2026.02.11/run1/snapshots/snapshot_100000.pt \
    --results-dir ./results_100k

# Test a 500k frames
./run_encoder_test.sh exp_local/2026.02.11/run1/snapshots/snapshot_500000.pt \
    --results-dir ./results_500k \
    --skip-sampling  # Riusa gli stessi episodi

# Test a 1M frames
./run_encoder_test.sh exp_local/2026.02.11/run1/snapshots/snapshot_1000000.pt \
    --results-dir ./results_1M \
    --skip-sampling
```

### Comparare diversi run

```bash
# Run 1
./run_encoder_test.sh exp_local/2026.02.11/run1/snapshot.pt \
    --results-dir ./comparison/run1

# Run 2 (stesso seed, diversi hyperparams)
./run_encoder_test.sh exp_local/2026.02.11/run2/snapshot.pt \
    --results-dir ./comparison/run2 \
    --skip-sampling  # Usa stessi episodi per fair comparison

# Run 3
./run_encoder_test.sh exp_local/2026.02.11/run3/snapshot.pt \
    --results-dir ./comparison/run3 \
    --skip-sampling
```

### Solo valutazioni specifiche

**Solo sampling:**

```bash
python sample_salient_episodes_pong.py \
    --save_dir ./episodes_pong \
    --num_episodes 10  # Più episodi per statistiche migliori
```

**Solo t-SNE:**

```python
# Modifica test_encoder_embeddings_pong.py per commentare altre valutazioni
from test_encoder_embeddings_pong import EncoderTester

tester = EncoderTester('snapshot.pt')
episodes = tester.load_episodes('./episodes')

# Solo t-SNE per traiettoria
tester.plot_tsne_per_trajectory(episodes, './tsne_only')
```

**Solo alignment/uniformity:**

```bash
python advanced_encoder_eval.py \
    --snapshot snapshot.pt \
    --episodes_dir ./episodes \
    --save_dir ./alignment_only \
    --num_samples 2000  # Più campioni per stima migliore
```

## Output Attesi

### Risultati Standard

```
encoder_test_results/
├── tsne_per_trajectory/
│   ├── tsne_ball_left_to_right_0.png
│   ├── tsne_ball_right_to_left_0.png
│   ├── tsne_player_wins_0.png
│   └── ...
├── tsne_all_trajectories.png
├── phi_vs_projected_psi.png
├── temporal_consistency.png
└── embedding_norms.png
```

### Risultati Avanzati

```
encoder_advanced_results/
├── alignment_uniformity.npy
├── linear_probe_confusion_matrix.png
├── intrinsic_dimension.png
└── all_advanced_results.npy
```

## Troubleshooting

### "CUDA out of memory"

```bash
# Usa CPU
./run_encoder_test.sh snapshot.pt --device cpu

# Oppure riduci batch size nel codice
# In test_encoder_embeddings_pong.py:
# def encode_observations(self, observations, batch_size=64):  # era 128
```

### "No episodes found"

```bash
# Assicurati di aver eseguito il sampling
ls -lh ./salient_episodes_pong/

# Se vuoto, esegui:
python sample_salient_episodes_pong.py
```

### "Encoder test troppo lento"

```bash
# Riduci num_episodes
./run_encoder_test.sh snapshot.pt --num-episodes 3

# Oppure riduci num_samples in advanced_encoder_eval.py
python advanced_encoder_eval.py --snapshot snapshot.pt --num_samples 500
```

### Episodi di un tipo non trovati

Se alcuni tipi di episodi sono difficili da campionare (es: mixed), puoi:

1. Aumentare `max_attempts` in `sample_salient_episodes_pong.py`
2. Ridurre `num_episodes_per_type`
3. Modificare la logica di classificazione episodi

## Tips & Tricks

### Velocizzare il sampling

```python
# In sample_salient_episodes_pong.py, riduci max_steps per episodio:
episode_data, episode_info = self.run_episode(max_steps=2000)  # era 5000
```

### Migliorare la qualità del sampling

```python
# Usa una politica migliore invece di random
# In sample_salient_episodes_pong.py:

def run_episode(self, max_steps=10000, policy='pretrained'):
    if policy == 'pretrained':
        # Carica politica addestrata
        action = loaded_policy(obs)
```

### Analisi su sottoinsieme di episodi

```python
# In test_encoder_embeddings_pong.py:

# Filtra solo alcuni tipi
episodes_filtered = {
    k: v for k, v in episodes_data.items() 
    if 'player_wins' in k or 'player_loses' in k
}

tester.plot_tsne_all_trajectories(episodes_filtered, save_dir)
```

### Esportare metriche per plot esterni

```python
import numpy as np

# Carica risultati
results = np.load('encoder_advanced_results/all_advanced_results.npy', allow_pickle=True).item()

print(f"Test Accuracy: {results['linear_probe']['test_acc']:.2%}")
print(f"Alignment: {results['alignment_uniformity']['alignment']:.4f}")
print(f"Intrinsic Dim: {results['intrinsic_dimension']['intrinsic_dim_95']}")
```

## Batch Testing su Multiple Seeds

```bash
#!/bin/bash
# test_all_seeds.sh

SEEDS=(1 2 3 4 5)
BASE_DIR="exp_local/2026.02.11"

for seed in "${SEEDS[@]}"; do
    SNAPSHOT="${BASE_DIR}/seed${seed}/snapshots/snapshot_1000000.pt"
    
    if [ -f "$SNAPSHOT" ]; then
        echo "Testing seed ${seed}..."
        ./run_encoder_test.sh "$SNAPSHOT" \
            --results-dir "./results_seed${seed}" \
            --skip-sampling  # Usa stessi episodi
    else
        echo "Snapshot not found: $SNAPSHOT"
    fi
done

echo "All seeds tested!"
```

## Visualizzare Risultati

```bash
# Apri tutti i plot
eog encoder_test_results/*.png

# Oppure crea un report HTML
python -c "
import base64
from pathlib import Path

html = '<html><body><h1>Encoder Test Results</h1>'
for img in Path('encoder_test_results').glob('*.png'):
    html += f'<h2>{img.stem}</h2><img src=\"{img}\" width=\"800\"><br><br>'
html += '</body></html>'

Path('report.html').write_text(html)
print('Report salvato: report.html')
"

# Apri nel browser
firefox report.html
```

---

Per maggiori dettagli, vedi `README_ENCODER_TESTING.md`
