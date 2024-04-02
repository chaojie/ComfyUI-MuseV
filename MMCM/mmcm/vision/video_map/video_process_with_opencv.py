import numpy as np
import cv2

import logging

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class VideoClipOperator(object):
    def __init__(self) -> None:
        pass

    def __call__(self, *args, **kwds):
        pass


class VSELength(VideoClipOperator):
    def __init__(self, time_start, duration, target, change_length_func) -> None:
        self.time_start = time_start
        self.duration = duration
        self.target = target
        self.change_length_func = change_length_func

    def __call__(self, cap, width, height, frame_count, fps):
        start_frame_idx = int(self.time_start * fps)
        src_frames_num = int(self.duration * fps)
        dst_frames_num = int(self.target * fps)
        # The first argument of cap.set(), number 2 defines that parameter for setting the frame selection.
        # Number 2 defines flag CV_CAP_PROP_POS_FRAMES which is a 0-based index of the frame to be decoded/captured next.
        # The second argument defines the frame number in range 0.0-1.0
        cap.set(2, start_frame_idx)
        src_frames = []
        for i in range(src_frames_num):
            frame = cap.read()
            src_frames.append(frame)
        src_frames = np.concatenate(src_frames, axis=0)
        # Read the next frame from the video.
        frames = self.change_length_func(src_frames, src_frames_num, dst_frames_num)
        return frames


class EditedVideoWriter(object):
    """do operators to videoclip

    Args:
        operators ([[VideoClipOperator,VideoClipOperator], [VideoClipOperator]]):
    """

    def __init__(self, operators):
        self.operators = operators

    def __call__(self, video, out):
        """
        1. open out path
        2. do operator to video, return edited video clip
        3. save

        Args:
            video (_type_): _description_
            out (_type_): _description_
        """
        cap = cv2.VideoCapture(video)
        # Check if camera opened successfully
        if cap.isOpened() == False:
            logger.error("Error opening video stream or file")

        out = cv2.VideoWriter(
            out,
            cv2.VideoWriter_fourcc("M", "J", "P", "G"),
            10,
            (self.width, self.height),
        )
        # float `width`
        for clip_operator in self.operators:
            frames = clip_operator(
                width=cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH),
                height=cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT),
                frame_count=cap.get(cv2.CAP_PROP_FRAME_COUNT),
                fps=cap.get(cv2.cv.CV_CAP_PROP_FPS),
            )
            for frame in frames:
                out.write(frame)
        out.release()
