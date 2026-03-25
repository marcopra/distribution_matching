import cv2
import imageio
import numpy as np
import wandb


def _format_video_frame(frame):
    frame = np.asarray(frame)
    if frame.ndim == 2:
        frame = frame[..., None]
    if frame.ndim != 3:
        raise ValueError(f"Expected 2D or 3D frame, got shape {frame.shape}")
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    return frame.astype(np.uint8)


class VideoRecorder:
    def __init__(self,
                 root_dir,
                 render_size=256,
                 fps=20,
                 camera_id=0,
                 use_wandb=False):
        if root_dir is not None:
            self.save_dir = root_dir / 'eval_video'
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.frames = []
        self.camera_id = camera_id
        self.use_wandb = use_wandb

    def init(self, env, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(env)

    def record(self, env):
        if self.enabled:
            if hasattr(env, 'physics'):
                frame = env.physics.render(height=self.render_size,
                                           width=self.render_size,
                                           camera_id=self.camera_id)
            else:
                frame = env.render()
            self.frames.append(_format_video_frame(frame))

    def log_to_wandb(self):
        frames = np.transpose(np.array(self.frames), (0, 3, 1, 2))
        fps, skip = 6, 8
        wandb.log({
            'eval/video':
            wandb.Video(frames[::skip, :, ::2, ::2], fps=fps, format="gif")
        })

    def save(self, file_name):
        if self.enabled:
            if self.use_wandb:
                self.log_to_wandb()
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)


class TrainVideoRecorder:
    def __init__(self,
                 root_dir,
                 render_size=256,
                 fps=20,
                 camera_id=0,
                 is_training_sample=True,
                 grayscale=False,
                 use_wandb=False):
        if root_dir is not None:
            self.save_dir = root_dir / 'train_video'
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.frames = []
        self.camera_id = camera_id
        self.use_wandb = use_wandb
        self.is_training_sample = is_training_sample
        self.grayscale = grayscale

    def init(self, obs, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(obs)

    def record(self, obs):
        if self.enabled:
            if self.is_training_sample:
                channels = 1 if self.grayscale else 3
                frame = cv2.resize(
                    obs[-channels:].transpose(1, 2, 0),
                    dsize=(self.render_size, self.render_size),
                    interpolation=cv2.INTER_CUBIC
                )
            else:
                frame = obs
            self.frames.append(_format_video_frame(frame))

    def log_to_wandb(self):
        frames = np.transpose(np.array(self.frames), (0, 3, 1, 2))
        fps, skip = 6, 8
        wandb.log({
            'train/video':
            wandb.Video(frames[::skip, :, ::2, ::2], fps=fps, format="gif")
        })

    def save(self, file_name):
        if self.enabled:
            if self.use_wandb:
                self.log_to_wandb()
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)
