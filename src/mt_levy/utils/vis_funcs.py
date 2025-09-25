import numpy as np
import imageio


__all__ = ['create_mp4_from_ndarray']


def create_mp4_from_ndarray(frames: np.ndarray, path: str, fps: int = 80, vertical_flip: bool = False):
    '''
    Inputs:
        frames: np.ndarray of shape (num_frames, height, width, 3)
        path: str, path to save the video
    '''
    if vertical_flip:
        frames = np.flip(frames, axis=1)  # flip the frames vertically
    imageio.mimsave(path, frames, fps=fps)
