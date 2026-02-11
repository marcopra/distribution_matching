"""
Script per campionare momenti salienti da Pong.
Rileva eventi specifici e salva solo brevi clip (finestre temporali) attorno a questi momenti.
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
import gym_env
from collections import deque
import cv2
from tqdm import tqdm
import argparse
from typing import Dict, List, Tuple, Optional


class CircularBuffer:
    """Buffer circolare per mantenere storia recente di osservazioni/frame"""
    
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.observations = deque(maxlen=maxlen)
        self.actions = deque(maxlen=maxlen)
        self.rewards = deque(maxlen=maxlen)
        self.frames = deque(maxlen=maxlen)
        self.step_indices = deque(maxlen=maxlen)
        
    def add(self, obs, action, reward, frame, step_idx):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.frames.append(frame)
        self.step_indices.append(step_idx)
    
    def get_window(self, center_idx, window_size):
        """Estrae una finestra di dati centrata su center_idx"""
        # Trova l'indice nel buffer corrispondente al center_idx
        try:
            buffer_idx = list(self.step_indices).index(center_idx)
        except ValueError:
            return None
        
        # Calcola range della finestra
        half_window = window_size // 2
        start = max(0, buffer_idx - half_window)
        end = min(len(self.observations), buffer_idx + half_window + 1)
        
        return {
            'observations': list(self.observations)[start:end],
            'actions': list(self.actions)[start:end],
            'rewards': list(self.rewards)[start:end],
            'frames': list(self.frames)[start:end],
        }
    
    def __len__(self):
        return len(self.observations)


class SalientMomentDetector:
    """Rileva momenti salienti specifici in Pong"""
    
    def __init__(self):
        self.prev_frame = None
        self.prev_reward = 0
        self.ball_position_history = deque(maxlen=10)
        
    def detect_events(self, frame, reward, step_idx) -> List[Tuple[str, int]]:
        """
        Rileva eventi salienti nel frame corrente.
        
        Returns:
            Lista di tuple (event_type, step_idx)
        """
        events = []
        
        # Evento 1: Reward positivo/negativo (punto segnato/subito)
        if reward > 0 and self.prev_reward == 0:
            events.append(('point_scored', step_idx))
        elif reward < 0 and self.prev_reward == 0:
            events.append(('point_lost', step_idx))
        
        # Eventi 2 & 3: Direzione palla e rimbalzi
        if self.prev_frame is not None:
            ball_pos = self._detect_ball_position(frame)
            
            if ball_pos is not None:
                self.ball_position_history.append(ball_pos)
                
                # Rileva direzione se abbiamo abbastanza storia
                if len(self.ball_position_history) >= 5:
                    direction = self._detect_ball_direction()
                    if direction == 'left_to_right':
                        events.append(('ball_left_to_right', step_idx))
                    elif direction == 'right_to_left':
                        events.append(('ball_right_to_left', step_idx))
                    
                    # Rileva rimbalzi (cambio brusco di direzione verticale)
                    bounce = self._detect_bounce()
                    if bounce == 'player_paddle':
                        events.append(('bounce_player', step_idx))
                    elif bounce == 'opponent_paddle':
                        events.append(('bounce_opponent', step_idx))
        
        self.prev_frame = frame.copy()
        self.prev_reward = reward
        
        return events
    
    def _detect_ball_position(self, frame):
        """Rileva posizione approssimativa della palla"""
        # Converti in grayscale se necessario
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame
        
        height, width = gray.shape
        
        # Maschera per escludere i bordi (top 15%, bottom 10%)
        # Questo evita di rilevare i bordi bianchi di Pong
        mask = np.zeros_like(gray)
        top_margin = int(height * 0.15)
        bottom_margin = int(height * 0.90)
        mask[top_margin:bottom_margin, :] = 255
        
        # La palla è tipicamente l'oggetto più luminoso nella zona di gioco
        # Threshold ridotto da 200 a 150 per catturare meglio la palla
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # Applica mask per escludere bordi
        thresh = cv2.bitwise_and(thresh, mask)
        
        # Trova contorni
        # OpenCV 3.x ritorna (image, contours, hierarchy)
        # OpenCV 4.x ritorna (contours, hierarchy)
        contours_result = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Gestisci versioni diverse di OpenCV
        if len(contours_result) == 3:
            _, contours, _ = contours_result
        else:
            contours, _ = contours_result
        
        if len(contours) > 0:
            # Filtra contorni per dimensione (palla è piccola, 1-100 pixel^2)
            valid_contours = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 0 < area < 100:  # Palla è piccola
                    valid_contours.append(cnt)
            
            if len(valid_contours) > 0:
                # Trova il contorno più piccolo tra quelli validi
                smallest = min(valid_contours, key=cv2.contourArea)
                M = cv2.moments(smallest)
                
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    return (cx, cy)
        
        return None
    
    def _detect_ball_direction(self):
        """Rileva direzione orizzontale della palla"""
        if len(self.ball_position_history) < 5:
            return None
        
        positions = list(self.ball_position_history)
        
        # Conta solo posizioni valide (non None)
        valid_positions = [p for p in positions if p is not None]
        if len(valid_positions) < 3:
            return None
        
        # Calcola velocità media orizzontale
        dx_sum = 0
        count = 0
        for i in range(1, len(valid_positions)):
            dx_sum += valid_positions[i][0] - valid_positions[i-1][0]
            count += 1
        
        if count > 0:
            avg_dx = dx_sum / count
            if avg_dx > 2:  # Soglia ridotta
                return 'left_to_right'
            elif avg_dx < -2:
                return 'right_to_left'
        
        return None
    
    def _detect_bounce(self):
        """Rileva rimbalzi sulla barra del giocatore o avversario"""
        if len(self.ball_position_history) < 5:
            return None
        
        positions = list(self.ball_position_history)
        
        # Filtra None
        valid_positions = [p for p in positions if p is not None]
        if len(valid_positions) < 4:
            return None
        
        # Verifica cambio direzione verticale
        recent_dy = []
        for i in range(1, len(valid_positions)):
            recent_dy.append(valid_positions[i][1] - valid_positions[i-1][1])
        
        if len(recent_dy) >= 3:
            # Rimbalzo = cambio di segno della velocità verticale
            # Controlla se c'è un cambio di direzione
            sign_changes = 0
            for i in range(1, len(recent_dy)):
                if recent_dy[i] * recent_dy[i-1] < 0:
                    sign_changes += 1
            
            if sign_changes > 0:
                # Determina se è rimbalzo del player o opponent
                last_pos = valid_positions[-1]
                # Player è in basso, opponent in alto (tipicamente)
                frame_height = 84  # Assumendo resolution standard
                if last_pos[1] > frame_height * 0.65:
                    return 'player_paddle'
                elif last_pos[1] < frame_height * 0.35:
                    return 'opponent_paddle'
        
        return None


class SalientEpisodeSampler:
    """Campiona clip salienti (brevi finestre temporali) da Pong"""
    
    def __init__(self, env, save_dir='./salient_episodes_pong', window_size=60, buffer_size=200):
        """
        Args:
            env: Environment Pong
            save_dir: Directory dove salvare le clip
            window_size: Dimensione della finestra (in step) da salvare attorno all'evento
            buffer_size: Dimensione del buffer circolare per mantenere storia recente
        """
        self.env = env
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        self.window_size = window_size
        self.buffer = CircularBuffer(maxlen=buffer_size)
        self.detector = SalientMomentDetector()
        
    def save_clip(self, clip_data, clip_name):
        """Salva clip in formato .npz e video .mp4"""
        
        if len(clip_data['observations']) == 0:
            print(f"⚠️  Clip vuota, skip: {clip_name}")
            return
        
        # Salva dati clip
        npz_path = self.save_dir / f'{clip_name}.npz'
        np.savez_compressed(
            npz_path,
            observations=np.array(clip_data['observations']),
            actions=np.array(clip_data['actions']),
            rewards=np.array(clip_data['rewards']),
            frames=np.array(clip_data['frames'])
        )
        
        # Salva video
        video_path = self.save_dir / f'{clip_name}.mp4'
        self._save_video(clip_data['frames'], video_path)
        
        print(f"  ✓ Salvata clip: {clip_name} ({len(clip_data['observations'])} frames)")
        
    def _save_video(self, frames, video_path, fps=30):
        """Salva frames come video mp4"""
        if len(frames) == 0:
            return
            
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        
        for frame in frames:
            # Converti da RGB a BGR per OpenCV
            if len(frame.shape) == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            video_writer.write(frame_bgr)
        
        video_writer.release()
    
    def run_and_collect_clips(self, target_clips_per_type=5, max_game_steps=50000):
        """
        Esegue il gioco e raccoglie clip brevi attorno a momenti salienti.
        
        Args:
            target_clips_per_type: Numero di clip da raccogliere per ogni tipo di evento
            max_game_steps: Numero massimo di step di gioco da eseguire
            
        Returns:
            Dict con clip raccolte per tipo
        """
        
        # Dizionario per conteggiare clip raccolte per tipo
        clip_counts = {
            'ball_left_to_right': 0,
            'ball_right_to_left': 0,
            'point_scored': 0,
            'point_lost': 0,
            'bounce_player': 0,
            'bounce_opponent': 0,
        }
        
        # Set per tracciare eventi già salvati (evitiamo duplicati vicini)
        recent_events = deque(maxlen=50)
        
        print("\n" + "=" * 60)
        print("CAMPIONAMENTO CLIP SALIENTI DA PONG")
        print("=" * 60)
        print(f"Target: {target_clips_per_type} clip per tipo di evento")
        print(f"Dimensione finestra: {self.window_size} frame")
        print(f"Max game steps: {max_game_steps}")
        print("=" * 60)
        
        # Reset env e inizializza
        time_step = self.env.reset()
        step_idx = 0
        prev_action = 0
        
        # Progress bar
        total_target_clips = target_clips_per_type * len(clip_counts)
        pbar = tqdm(total=total_target_clips, desc="Clip raccolte")
        
        while step_idx < max_game_steps:
            # Controlla se abbiamo completato la raccolta
            all_complete = all(count >= target_clips_per_type for count in clip_counts.values())
            if all_complete:
                print("\n✓ Tutte le clip target raccolte!")
                break
            
            # Reset se episodio terminato
            if time_step.last():
                time_step = self.env.reset()
                self.detector = SalientMomentDetector()  # Reset detector
                self.buffer = CircularBuffer(maxlen=self.buffer.maxlen)  # Reset buffer
                continue
            
            # Azione random
            action = self.env.action_space.sample()
            
            # Step environment
            time_step = self.env.step(action)
            frame = self.env.render()
            
            # Aggiungi al buffer
            self.buffer.add(
                obs=time_step.observation,
                action=prev_action,
                reward=time_step.reward,
                frame=frame,
                step_idx=step_idx
            )
            
            # Rileva eventi salienti
            events = self.detector.detect_events(frame, time_step.reward, step_idx)
            
            # Salva clip per ogni evento rilevato
            for event_type, event_step_idx in events:
                # Controlla se abbiamo già raccolto abbastanza clip di questo tipo
                if clip_counts.get(event_type, 0) >= target_clips_per_type:
                    continue
                
                # Evita eventi duplicati troppo vicini
                if event_step_idx in recent_events:
                    continue
                
                # Estrai finestra attorno all'evento
                clip_data = self.buffer.get_window(event_step_idx, self.window_size)
                
                if clip_data is not None and len(clip_data['observations']) > 0:
                    # Salva clip
                    clip_name = f"{event_type}_{clip_counts[event_type]}"
                    self.save_clip(clip_data, clip_name)
                    
                    # Aggiorna contatori
                    clip_counts[event_type] += 1
                    recent_events.append(event_step_idx)
                    pbar.update(1)
            
            prev_action = action
            step_idx += 1
        
        pbar.close()
        
        # Riepilogo finale
        print("\n" + "=" * 60)
        print("RIEPILOGO CAMPIONAMENTO")
        print("=" * 60)
        for event_type, count in sorted(clip_counts.items()):
            status = "✓" if count >= target_clips_per_type else "⚠"
            print(f"{status} {event_type:25s}: {count:2d}/{target_clips_per_type} clip")
        print("=" * 60)
        
        return clip_counts


def main():
    parser = argparse.ArgumentParser(description='Campiona clip salienti da Pong')
    parser.add_argument('--save_dir', type=str, default='./data_salient_episodes_pong',
                       help='Directory dove salvare le clip')
    parser.add_argument('--num_clips', '--num_episodes', type=int, default=5,
                       help='Numero di clip per tipo da campionare')
    parser.add_argument('--window_size', type=int, default=60,
                       help='Dimensione finestra temporale (in frame) attorno all\'evento')
    parser.add_argument('--buffer_size', type=int, default=200,
                       help='Dimensione buffer circolare per storia recente')
    parser.add_argument('--max_steps', type=int, default=50000,
                       help='Numero massimo di step di gioco')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--resolution', type=int, default=84,
                       help='Risoluzione immagini')
    parser.add_argument('--frame_stack', type=int, default=3,
                       help='Numero di frame da stackare')
    parser.add_argument('--action_repeat', type=int, default=4,
                       help='Action repeat')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CAMPIONAMENTO CLIP SALIENTI - PONG")
    print("=" * 60)
    print(f"Environment: PongNoFrameskip-v4")
    print(f"Resolution: {args.resolution}")
    print(f"Frame stack: {args.frame_stack}")
    print(f"Action repeat: {args.action_repeat}")
    print(f"Window size: {args.window_size} frame")
    print(f"Buffer size: {args.buffer_size} frame")
    print(f"Seed: {args.seed}")
    print(f"Save dir: {args.save_dir}")
    print("=" * 60)
    
    # Crea environment come in gym_env.py
    env = gym_env.make(
        name='PongNoFrameskip-v4',
        obs_type='pixels',
        frame_stack=args.frame_stack,
        action_repeat=args.action_repeat,
        seed=args.seed,
        resolution=args.resolution,
        random_init=False,
        randomize_goal=False,
        url=False,
        render_mode='rgb_array'
    )
    
    # Crea sampler e campiona clip
    sampler = SalientEpisodeSampler(
        env, 
        save_dir=args.save_dir,
        window_size=args.window_size,
        buffer_size=args.buffer_size
    )
    
    clip_counts = sampler.run_and_collect_clips(
        target_clips_per_type=args.num_clips,
        max_game_steps=args.max_steps
    )
    
    print(f"\n✓ Clip salvate in: {args.save_dir}")
    print("  Ogni clip include:")
    print("    - File .npz con observations, actions, rewards, frames")
    print("    - File .mp4 con video della clip")
    print(f"\n  Clip raccolte: {sum(clip_counts.values())} totali")


if __name__ == '__main__':
    main()
