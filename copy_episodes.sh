#!/bin/bash
# filepath: copy_episodes.sh

SRC_DIR="$1"
DST_DIR="$2"
MAX_TOTAL_LENGTH="${3:-100000}"

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