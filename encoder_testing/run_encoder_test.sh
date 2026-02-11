#!/bin/bash

# Script per eseguire il workflow completo di testing dell'encoder su Pong
# Usage: ./run_encoder_test.sh [snapshot_path]

set -e  # Exit on error

# Directory dello script (compatibile con sh)
SCRIPT_DIR="$(cd "$(dirname "$0")" >/dev/null 2>&1 && pwd)"

# Colori per output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parametri di default
EPISODES_DIR="./data_salient_episodes_pong"
RESULTS_DIR="./encoder_test_results"
NUM_EPISODES=5
RESOLUTION=84
FRAME_STACK=3
ACTION_REPEAT=4
SEED=42
DEVICE="cuda"

# Funzione per stampare con colore
print_step() {
    echo -e "${BLUE}======================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}======================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Banner
echo -e "${GREEN}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║       ENCODER TESTING WORKFLOW - PONG                      ║"
echo "║         Testing Contrastive Learning Embeddings            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Controlla argomenti
if [ "$#" -eq 0 ]; then
    print_error "Snapshot path richiesto!"
    echo "Usage: $0 <snapshot_path> [options]"
    echo ""
    echo "Options:"
    echo "  --episodes-dir DIR     Directory per episodi (default: $EPISODES_DIR)"
    echo "  --results-dir DIR      Directory per risultati (default: $RESULTS_DIR)"
    echo "  --num-episodes N       Numero episodi per tipo (default: $NUM_EPISODES)"
    echo "  --skip-sampling        Salta il sampling (usa episodi esistenti)"
    echo "  --device DEVICE        Device (cuda/cpu, default: $DEVICE)"
    echo ""
    echo "Example:"
    echo "  $0 ./exp_local/2026.02.11/snapshot.pt"
    echo "  $0 ./exp_local/2026.02.11/snapshot.pt --skip-sampling"
    exit 1
fi

SNAPSHOT_PATH=$1
shift

# Parse opzioni
SKIP_SAMPLING=false

while [ $# -gt 0 ]; do
    case $1 in
        --episodes-dir)
            EPISODES_DIR="$2"
            shift 2
            ;;
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --num-episodes)
            NUM_EPISODES="$2"
            shift 2
            ;;
        --skip-sampling)
            SKIP_SAMPLING=true
            shift
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        *)
            print_error "Opzione sconosciuta: $1"
            exit 1
            ;;
    esac
done

# Verifica snapshot
if [ ! -f "$SNAPSHOT_PATH" ]; then
    print_error "Snapshot non trovato: $SNAPSHOT_PATH"
    exit 1
fi

print_success "Snapshot trovato: $SNAPSHOT_PATH"

# Configurazione
echo ""
echo "Configurazione:"
echo "  Snapshot:      $SNAPSHOT_PATH"
echo "  Episodes dir:  $EPISODES_DIR"
echo "  Results dir:   $RESULTS_DIR"
echo "  Num episodes:  $NUM_EPISODES"
echo "  Device:        $DEVICE"
echo "  Skip sampling: $SKIP_SAMPLING"
echo ""

# Attendi conferma
printf "Procedere? [Y/n] "
read REPLY
if [ "$REPLY" != "Y" ] && [ "$REPLY" != "y" ] && [ -n "$REPLY" ]; then
    print_warning "Operazione annullata"
    exit 0
fi

# STEP 1: Campionamento episodi (se non saltato)
if [ "$SKIP_SAMPLING" = false ]; then
    print_step "STEP 1/3: Campionamento Episodi Salienti"
    
    if [ -d "$EPISODES_DIR" ] && [ "$(ls -A $EPISODES_DIR)" ]; then
        print_warning "Directory episodi già esistente con contenuto"
        printf "Sovrascrivere? [y/N] "
        read REPLY
        if [ "$REPLY" = "Y" ] || [ "$REPLY" = "y" ]; then
            rm -rf "$EPISODES_DIR"
            print_success "Directory ripulita"
        else
            print_warning "Usando episodi esistenti"
        fi
    fi
    
    if [ ! -d "$EPISODES_DIR" ] || [ ! "$(ls -A $EPISODES_DIR)" ]; then
        python "$SCRIPT_DIR/sample_salient_episodes_pong.py" \
            --save_dir "$EPISODES_DIR" \
            --num_episodes "$NUM_EPISODES" \
            --seed "$SEED" \
            --resolution "$RESOLUTION" \
            --frame_stack "$FRAME_STACK" \
            --action_repeat "$ACTION_REPEAT"
        
        if [ $? -eq 0 ]; then
            print_success "Episodi campionati con successo"
        else
            print_error "Errore nel campionamento"
            exit 1
        fi
    fi
else
    print_step "STEP 1/3: Campionamento (Saltato)"
    
    # Verifica che esistano episodi
    if [ ! -d "$EPISODES_DIR" ] || [ ! "$(ls -A $EPISODES_DIR)" ]; then
        print_error "Directory episodi non trovata o vuota: $EPISODES_DIR"
        exit 1
    fi
    
    print_success "Usando episodi esistenti in $EPISODES_DIR"
fi

# STEP 2: Ispezione episodi
print_step "STEP 2/3: Ispezione Episodi"

python "$SCRIPT_DIR/inspect_salient_episodes.py" --episodes_dir "$EPISODES_DIR"

if [ $? -eq 0 ]; then
    print_success "Ispezione completata"
else
    print_error "Errore nell'ispezione"
    exit 1
fi

echo ""
printf "Procedere con il testing dell'encoder? [Y/n] "
read REPLY
if [ "$REPLY" != "Y" ] && [ "$REPLY" != "y" ] && [ -n "$REPLY" ]; then
    print_warning "Testing annullato"
    exit 0
fi

# STEP 3: Testing encoder
print_step "STEP 3/3: Testing Encoder e Generazione Visualizzazioni"

# Crea directory risultati
mkdir -p "$RESULTS_DIR"

python "$SCRIPT_DIR/test_encoder_embeddings_pong.py" \
    --snapshot "$SNAPSHOT_PATH" \
    --episodes_dir "$EPISODES_DIR" \
    --save_dir "$RESULTS_DIR" \
    --device "$DEVICE"

if [ $? -eq 0 ]; then
    print_success "Testing completato!"
else
    print_error "Errore nel testing"
    exit 1
fi

# Riepilogo finale
echo ""
echo -e "${GREEN}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                  WORKFLOW COMPLETATO                       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

print_success "Episodi salvati in: $EPISODES_DIR"
print_success "Risultati salvati in: $RESULTS_DIR"

echo ""
echo "Visualizzazioni generate:"
echo "  📊 t-SNE per traiettoria:      $RESULTS_DIR/tsne_per_trajectory/"
echo "  📊 t-SNE tutte traiettorie:    $RESULTS_DIR/tsne_all_trajectories.png"
echo "  📊 φ(s') vs W·ψ(s,a):          $RESULTS_DIR/phi_vs_projected_psi.png"
echo "  📊 Consistenza temporale:      $RESULTS_DIR/temporal_consistency.png"
echo "  📊 Norme embeddings:           $RESULTS_DIR/embedding_norms.png"

echo ""
echo "Per vedere i risultati:"
echo "  cd $RESULTS_DIR"
echo "  eog *.png  # oppure qualsiasi image viewer"

# Opzione per aprire automaticamente i risultati
if command -v eog >/dev/null 2>&1; then
    echo ""
    printf "Aprire i risultati con eog? [y/N] "
    read REPLY
    if [ "$REPLY" = "Y" ] || [ "$REPLY" = "y" ]; then
        eog "$RESULTS_DIR"/*.png &
    fi
fi

print_success "Workflow completato con successo!"
