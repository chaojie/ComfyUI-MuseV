from typing import Sequence

from numpy import ndarray
import cv2
try:
    import imageio
except ImportError:
    imageio = None


def create_video(frames: Sequence[ndarray], out: str, fourcc: int, fps: int,
                 size: tuple) -> None:
    """Create a video to save the optical flow.
    ## from mmflow
    Args:
        frames (list, tuple): Image frames.
        out (str): The output file to save visualized flow map.
        fourcc (int): Code of codec used to compress the frames.
        fps (int):      Framerate of the created video stream.
        size (tuple): Size of the video frames.
    """
    # init video writer
    video_writer = cv2.VideoWriter(out, fourcc, fps, size, True)

    for frame in frames:
        video_writer.write(frame)
    video_writer.release()


def create_gif(frames: Sequence[ndarray],
               gif_name: str,
               duration: float = 0.1) -> None:
    """Create gif through imageio.
    ## from mmflow

    Args:
        frames (list[ndarray]): Image frames.
        gif_name (str): Saved gif name
        duration (int): Display interval (s). Default: 0.1.
    """
    frames_rgb = []
    for frame in frames:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_rgb.append(frame_rgb)
    if imageio is None:
        raise RuntimeError('imageio is not installed,'
                           'Please use “pip install imageio” to install')
    imageio.mimsave(gif_name, frames_rgb, 'GIF', duration=duration)
