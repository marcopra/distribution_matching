# Testing Encoder con Episodi Salienti - Pong

> **Nota**: Tutti gli script di testing sono nella cartella `encoder_testing/`. 
> Esegui i comandi dalla directory principale del progetto usando `encoder_testing/script.py` 
> oppure entra nella cartella: `cd encoder_testing` e poi esegui gli script.

Questo set di script permette di testare la qualità dell'encoder dell'algoritmo usando episodi salienti da Pong.

## 📁 File Creati

1. **`sample_salient_episodes_pong.py`** - Campiona episodi salienti da Pong
2. **`test_encoder_embeddings_pong.py`** - Testa l'encoder e genera visualizzazioni
3. **`inspect_salient_episodes.py`** - Utility per ispezionare episodi salvati

## 🚀 Workflow Completo

### Step 1: Campiona Episodi Salienti

Esegui lo script per campionare episodi da Pong:

```bash
python sample_salient_episodes_pong.py \
    --save_dir ./salient_episodes_pong \
    --num_episodes 5 \
    --seed 42 \
    --resolution 84 \
    --frame_stack 3 \
    --action_repeat 4
```

**Parametri:**
- `--save_dir`: Directory dove salvare gli episodi (default: `./salient_episodes_pong`)
- `--num_episodes`: Numero di episodi per tipo da campionare (default: 5)
- `--seed`: Random seed (default: 42)
- `--resolution`: Risoluzione immagini (default: 84)
- `--frame_stack`: Numero di frame da stackare (default: 3)
- `--action_repeat`: Action repeat (default: 4)

**Tipi di episodi campionati:**
1. `ball_left_to_right_*` - Pallina che va da sinistra a destra
2. `ball_right_to_left_*` - Pallina che va da destra a sinistra
3. `player_wins_*` - Episodi con reward positivo (+1, punto segnato)
4. `player_loses_*` - Episodi con reward negativo (-1, punto subito)
5. `mixed_*` - Episodi con sia reward positivi che negativi

**Output:**
- File `.npz` con: observations, actions, rewards, dones, frames
- File `.mp4` con video dell'episodio

### Step 2: Ispeziona Episodi (Opzionale)

Verifica che gli episodi siano stati salvati correttamente:

```bash
python inspect_salient_episodes.py --episodes_dir ./salient_episodes_pong
```

Questo mostra:
- Numero di episodi per tipo
- Steps e reward per episodio
- Shape delle osservazioni
- Verifica consistenza dei dati

### Step 3: Testa l'Encoder

Carica uno snapshot dell'agente e testa l'encoder:

```bash
python test_encoder_embeddings_pong.py \
    --snapshot ./exp_local/path/to/snapshot.pt \
    --episodes_dir ./salient_episodes_pong \
    --save_dir ./encoder_test_results \
    --device cuda
```

**Parametri:**
- `--snapshot`: Path allo snapshot.pt dell'agente (richiesto)
- `--episodes_dir`: Directory con episodi salvati (default: `./salient_episodes_pong`)
- `--save_dir`: Directory dove salvare i plot (default: `./encoder_test_results`)
- `--device`: Device (cuda o cpu, default: cuda)

## 📊 Visualizzazioni Generate

Lo script di testing genera diverse visualizzazioni per valutare la qualità dell'encoder:

### 1. t-SNE per Traiettoria

**Directory:** `encoder_test_results/tsne_per_trajectory/`

Un plot t-SNE separato per ogni episodio:
- Colore: gradiente temporale (dal primo all'ultimo step)
- Frecce: mostrano la direzione temporale
- **Obiettivo**: Verificare se stati simili temporalmente sono vicini nello spazio embedding

### 2. t-SNE Tutte le Traiettorie

**File:** `encoder_test_results/tsne_all_trajectories.png`

Plot t-SNE con tutti gli episodi insieme:
- Ogni episodio ha un colore diverso
- **Obiettivo**: Verificare se episodi dello stesso tipo si raggruppano

### 3. φ(s') vs W·ψ(s,a)

**File:** `encoder_test_results/phi_vs_projected_psi.png`

Quattro subplot che analizzano la relazione tra φ(s') e W·ψ(s,a):

a) **Cosine Similarity Distribution**
   - Distribuzione delle similarità coseno tra φ(s') e W·ψ(s,a)
   - **Ideale**: Valori vicini a 1 (perfettamente allineati)

b) **L2 Distance Distribution**
   - Distribuzione delle distanze L2
   - **Ideale**: Valori bassi (embeddings vicini)

c) **t-SNE Congiunto**
   - Visualizza φ(s') e W·ψ(s,a) nello stesso spazio
   - **Ideale**: I due tipi di embedding si sovrappongono

d) **Correlation: Dimension 0**
   - Scatter plot della correlazione sulla prima dimensione
   - Linea rossa y=x mostra correlazione perfetta
   - **Ideale**: Punti vicini alla linea y=x

**Interpretazione:**
- Se il contrastive learning funziona bene, φ(s') e W·ψ(s,a) dovrebbero essere vicini
- Alta cosine similarity (>0.8) indica buon allineamento
- Bassa L2 distance indica embeddings simili

### 4. Analisi Qualità Embeddings

**File:** `encoder_test_results/temporal_consistency.png`

Distribuzione delle distanze L2 tra stati consecutivi:
- **Ideale**: Distribuzione con coda verso valori bassi
- Stati consecutivi dovrebbero avere embeddings simili

**File:** `encoder_test_results/embedding_norms.png`

Distribuzione delle norme degli embeddings:
- **Obiettivo**: Verificare che le norme siano stabili
- Utile per identificare collapsed embeddings (tutti simili)

**Output Console:**

Distanze tra centroidi di tipi diversi di episodi:
- Episodi di tipo diverso dovrebbero avere centroidi distanti
- **Ideale**: Distanze elevate tra tipi diversi

## 💡 Suggerimenti per l'Interpretazione

### Encoder Funziona Bene Se:

1. **t-SNE mostra clustering**:
   - Episodi dello stesso tipo si raggruppano
   - Traiettorie temporali sono smooth (stati consecutivi vicini)

2. **φ(s') ≈ W·ψ(s,a)**:
   - Cosine similarity > 0.7
   - L2 distance bassa
   - Overlap nel t-SNE congiunto

3. **Consistenza temporale**:
   - Distanze tra stati consecutivi piccole e stabili
   - No salti improvvisi negli embeddings

4. **Separazione tra tipi**:
   - Episodi con eventi diversi (wins vs loses, left vs right) hanno centroidi distanti

### Encoder Ha Problemi Se:

1. **Random embeddings**:
   - No clustering nel t-SNE
   - Punti distribuiti uniformemente

2. **Collapsing**:
   - Tutti gli embeddings hanno norme molto simili
   - No diversità negli embeddings

3. **φ(s') ≠ W·ψ(s,a)**:
   - Cosine similarity bassa (<0.5)
   - No overlap nel t-SNE congiunto

## 📈 Altre Valutazioni Suggerite

### 1. Linear Probing

Addestra un classificatore lineare sugli embeddings per predire:
- Posizione della palla (sinistra/destra/centro)
- Se il giocatore ha appena segnato/subito un punto
- **Se gli embeddings sono buoni, un classificatore lineare dovrebbe funzionare bene**

```python
# Esempio: classificare direzione palla
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Etichetta episodi: 0=left_to_right, 1=right_to_left
labels = [...]  # Crea labels per gli embeddings
embeddings = [...]  # Tutti gli embeddings

X_train, X_test, y_train, y_test = train_test_split(embeddings, labels)
clf = LogisticRegression()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(f"Linear probe accuracy: {accuracy:.2%}")
```

### 2. Nearest Neighbor Analysis

Trova i k-nearest neighbors per ogni stato:
- Dovrebbero essere stati visualmente/semanticamente simili
- Verifica manualmente alcuni esempi

### 3. Intrinsic Dimension

Stima la dimensione intrinseca degli embeddings:
- Se troppo bassa → collapsing
- Se troppo alta → ridondanza

```python
from sklearn.decomposition import PCA

pca = PCA()
pca.fit(embeddings.numpy())

# Plot explained variance
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
```

### 4. Alignment e Uniformity

Metriche da "Understanding Contrastive Learning" (Wang & Isola, 2020):

**Alignment**: Quanto sono vicini gli embedding positivi
**Uniformity**: Quanto sono distribuiti uniformemente

```python
def alignment(x, y, alpha=2):
    return (x - y).norm(dim=1).pow(alpha).mean()

def uniformity(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

# Per φ(s') e W·ψ(s,a)
align = alignment(phi_next, projected_psi)
unif_phi = uniformity(phi_next)
unif_psi = uniformity(projected_psi)

print(f"Alignment: {align:.4f} (lower is better)")
print(f"Uniformity φ(s'): {unif_phi:.4f} (lower is better)")
print(f"Uniformity W·ψ(s,a): {unif_psi:.4f} (lower is better)")
```

### 5. Visualization dei Feature Importanti

Usa Grad-CAM o Integrated Gradients per vedere quali parti dell'immagine influenzano l'embedding:
- Dovrebbe focalizzarsi su palla e barrette
- Non su background o artefatti

## 🛠️ Troubleshooting

### Problema: "Nessun episodio trovato"

Verifica che lo script di sampling sia stato eseguito correttamente:
```bash
ls -lh ./salient_episodes_pong/
```

### Problema: "Agent non ha encoder"

Lo snapshot potrebbe essere di un agente diverso. Verifica:
```python
snapshot = torch.load('snapshot.pt')
print(type(snapshot['agent']))
print(dir(snapshot['agent']))
```

### Problema: "Out of Memory"

Riduci il batch size o il numero di campioni:
```python
# In test_encoder_embeddings_pong.py, modifica:
def plot_phi_vs_projected_psi(self, episodes_data, save_dir, num_samples=500):  # era 1000
```

### Problema: "t-SNE troppo lento"

Usa PCA invece di t-SNE per una visualizzazione più veloce:
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings_np)
```

## 📝 Note

- Gli episodi vengono salvati usando una politica random, quindi potrebbero essere necessari molti tentativi per campionare tutti i tipi
- Se alcuni tipi di episodi sono difficili da ottenere, riduci `num_episodes_per_type` o modifica la logica di classificazione
- I video .mp4 sono utili per verificare visivamente cosa succede negli episodi

## 🔗 File Correlati

- `agent/dist_matching_embedding_augmented_v2.py` - Implementazione dell'agente
- `gym_env.py` - Creazione dell'environment
- `pretrain.py` - Script di training

---

**Autore**: Script creati per testare encoder con contrastive learning su Pong  
**Data**: Febbraio 2026
