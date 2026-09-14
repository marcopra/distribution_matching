#!/bin/bash
# filepath: copy_episodes.sh

SRC_DIR="$1"
DST_DIR="$2"
MAX_TOTAL_LENGTH="${3:-1000000}"

if [[ -z "$SRC_DIR" || -z "$DST_DIR" ]]; then
  echo "Uso: $0 <cartella_sorgente> <cartella_destinazione> [max_total_length]"
  exit 1
fi

mkdir -p "$DST_DIR"

total_length=0

for file in $(ls "$SRC_DIR"/*.npz | sort); do
  # Estrai la lunghezza episodio dal nome file (ultimo campo, separato da _ e prima di .npz)
  filename=$(basename "$file")
  episode_length=$(echo "$filename" | awk -F'[_\.]' '{print $(NF-1)}')
  # Se non è un numero, salta
  if ! [[ "$episode_length" =~ ^[0-9]+$ ]]; then
    continue
  fi
  # Se superiamo la soglia, fermati
  if (( total_length + episode_length > MAX_TOTAL_LENGTH )); then
    break
  fi
  cp "$file" "$DST_DIR/"
  total_length=$((total_length + episode_length))
done

echo "Episodi copiati. Lunghezza totale: $total_length"

python tests/diagnostics/pointmaze/sweep_pointmaze_synthetic_pmd.py --workflow-dir tests/outputs/pointmaze/synthetic_workflow_l1_8k --dataset-dir tests/outputs/pointmaze/synthetic_workflow_8k/dataset --feature-dims 64 128 --feature-whitening pca --whitening-variance 0.99 --whitening-components 0 --whitening-epsilon 1e-5 --whitening-unit-trace --kernels gaussian --bandwidths none --bandwidth-mults 0.5 0.4 0.3 0.2 0.1 --lambda-regs 1e-2 1e-3 1e-4 1e-5 1e-7 --landmarks 7956 --pmd-steps 5 --etas 100 --sink 0.001 0.01 0.1 0.25 0.5 0.8 1.0 5.0 10.0 50.0 100.0 --eval-trajectories 50 --encode-batch-size 256  --device cuda
python tests/diagnostics/pointmaze/sweep_pointmaze_synthetic_pmd.py --workflow-dir tests/outputs/pointmaze/synthetic_workflow_l2_8k --dataset-dir tests/outputs/pointmaze/synthetic_workflow_8k/dataset --feature-dims 64 --feature-whitening pca --whitening-variance 0.99 --whitening-components 0 --whitening-epsilon 1e-5 --whitening-unit-trace --kernels gaussian --bandwidths none --bandwidth-mults 0.5 0.4 0.3 0.2 0.1 --lambda-regs 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7 --landmarks 7956 --pmd-steps 5 --etas 100 --sink 0.001 0.01 0.1 0.25 0.5 0.8 --eval-trajectories 50 --encode-batch-size 256  --device cuda

python tests/diagnostics/pointmaze/sweep_pointmaze_synthetic_pmd.py --workflow-dir tests/outputs/pointmaze/synthetic_workflow_l1_8k --dataset-dir tests/outputs/pointmaze/synthetic_workflow_8k/dataset --feature-dims 16 32 --feature-whitening pca --whitening-variance 0.99 --whitening-components 0 --whitening-epsilon 1e-5 --whitening-unit-trace --kernels gaussian --bandwidths none --bandwidth-mults 0.5 0.4 0.3 0.2 0.1 --lambda-regs 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7  --landmarks 7956 --pmd-steps 5 --etas 100 --sink 0.001 0.01 0.1 0.25 0.5 0.8 1.0 5.0 10.0 50.0 100.0 --eval-trajectories 50 --encode-batch-size 256  --device cuda
python tests/diagnostics/pointmaze/sweep_pointmaze_synthetic_pmd.py --workflow-dir tests/outputs/pointmaze/synthetic_workflow_l2_8k --dataset-dir tests/outputs/pointmaze/synthetic_workflow_8k/dataset --feature-dims 16 32 --feature-whitening pca --whitening-variance 0.99 --whitening-components 0 --whitening-epsilon 1e-5 --whitening-unit-trace --kernels gaussian --bandwidths none --bandwidth-mults 0.5 0.4 0.3 0.2 0.1 --lambda-regs 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7  --landmarks 7956 --pmd-steps 5 --etas 100 --sink 0.001 0.01 0.1 0.25 0.5 0.8 1.0 5.0 10.0 50.0 100.0 --eval-trajectories 50 --encode-batch-size 256  --device cuda


