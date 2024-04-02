import os
import itertools
from collections import namedtuple
from typing import Any, Iterator, List, Literal, Sequence
import math
from einops import rearrange

from moviepy.editor import VideoFileClip
import torchvision
from torch.utils.data.dataset import Dataset, IterableDataset
from PIL import Image
import cv2
import torch
import numpy as np

from ...utils.path_util import get_dir_file_map
from ...utils.itertools_util import generate_sample_idxs, overlap2step, step2overlap


VideoDatasetOutput = namedtuple("video_dataset_output", ["data", "index"])


def worker_init_fn(worker_id: int):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    overall_start = 0
    overall_end = len(dataset)
    # configure the dataset to only process the split workload
    per_worker = int(
        math.ceil((overall_end - overall_start) / float(worker_info.num_workers))
    )
    worker_id = worker_info.id
    dataset_start = overall_start + worker_id * per_worker
    dataset_end = min(overall_start + per_worker, overall_end)
    dataset.sample_indexs = dataset.sample_indexs[dataset_start:dataset_end]


class SequentialDataset(IterableDataset):
    def __init__(
        self,
        raw_datas,
        time_size: int,
        step: int,
        overlap: int = None,
        sample_rate: int = 1,
        drop_last: bool = False,
        max_num_per_batch: int = None,
        data_type: Literal["bgr", "rgb"] = "bgr",
        channels_order: str = "t h w c",
        sample_indexs: List[List[int]] = None,
    ) -> None:
        """_summary_

        Args:
            raw_datas (_type_): all original data
            time_size (int): frames number of a clip
            step (int): step of two windows
            overlap (int, optional): overlap of two windows. Defaults to None.
            sample_rate (int, optional): sample 1 evey sample_rate number. Defaults to 1.
            drop_last (bool, optional): whether drop the last if length of last batch < time_size. Defaults to False.
        """
        super().__init__()
        self.time_size = time_size
        if overlap is not None and step is None:
            step = overlap2step(overlap, time_size)
        if step is not None and overlap is None:
            overlap = step2overlap(step, time_size)
        self.overlap = overlap
        self.step = step
        self.sample_rate = sample_rate
        self.drop_last = drop_last
        self.raw_datas = raw_datas
        self.max_num_per_batch = max_num_per_batch
        if sample_indexs is not None:
            self.sample_indexs = sample_indexs
        else:
            self.generate_sample_idxs()
        self.current_pos = 0
        self.data_type = data_type
        self.channels_order = channels_order

    def generate_sample_idxs(
        self,
    ):
        self.sample_indexs = generate_sample_idxs(
            total=self.total_frames,
            window_size=self.time_size,
            step=self.step,
            sample_rate=self.sample_rate,
            drop_last=self.drop_last,
            max_num_per_window=self.max_num_per_batch,
        )

    def get_raw_datas(
        self,
    ):
        return self.raw_datas

    def get_raw_data(self, index: int):
        raise NotImplementedError

    def get_batch_raw_data(self, indexs: List[int]):
        datas = [self.get_raw_data(i) for i in indexs]
        datas = np.stack(datas, axis=0)
        return datas

    def __len__(self):
        return len(self.sample_indexs)

    def __iter__(self) -> Iterator[Any]:
        return self

    def __getitem__(self, index):
        sample_indexs = self.sample_indexs[index]
        data = self.get_batch_raw_data(sample_indexs)
        if self.channels_order != "t h w c":
            data = rearrange(data, "t h w c -> {}".format(self.channels_order))
        sample_indexs = np.array(sample_indexs)
        return VideoDatasetOutput(data, sample_indexs)

    def get_data(self, index):
        return self.__getitem__(index)

    def __next__(self):
        while self.current_pos < len(self.sample_indexs):
            data = self.get_data(self.current_pos)
            self.current_pos += 1
            return data
        self.current_pos = 0
        raise StopIteration

    def preview(self, clip):
        """show data clip,
        play for image, video, and print for str list

        Args:
            clip (_type_): _description_
        """
        raise NotImplementedError

    def close(self):
        """
        close file handle if subclass open file
        """
        raise NotImplementedError

    @property
    def fps(self):
        raise NotImplementedError

    @property
    def total_frames(self):
        raise NotImplementedError

    @property
    def duration(self):
        raise NotImplementedError

    @property
    def width(self):
        raise NotImplementedError

    @property
    def height(self):
        raise NotImplementedError


class ItemsSequentialDataset(SequentialDataset):
    def __init__(
        self,
        raw_datas: Sequence,
        time_size: int,
        step: int,
        overlap: int = None,
        sample_rate: int = 1,
        drop_last: bool = False,
        sample_indexs: List[List[int]] = None,
    ) -> None:
        super().__init__(
            raw_datas,
            time_size,
            step,
            overlap,
            sample_rate,
            drop_last,
            sample_indexs=sample_indexs,
        )

    def get_raw_data(self, index: int):
        return self.raw_datas[index]

    def prepare_raw_datas(self, raw_datas) -> Sequence:
        return raw_datas

    @property
    def total_frames(self):
        return len(self.raw_datas)


class ListSequentialDataset(ItemsSequentialDataset):
    def preview(self, clip):
        print(f"type is {self.__class__.__name__}, num is {len(clip)}")
        print(clip)


class ImagesSequentialDataset(ItemsSequentialDataset):
    def __init__(
        self,
        img_dir: Sequence,
        time_size: int,
        step: int,
        overlap: int = None,
        sample_rate: int = 1,
        drop_last: bool = False,
        data_type: Literal["bgr", "rgb"] = "bgr",
        channels_order: str = "t h w c",
        sample_indexs: List[List[int]] = None,
    ) -> None:
        self.imgs_path = sorted(get_dir_file_map(img_dir).values())
        super().__init__(
            self.imgs_path,
            time_size,
            step,
            overlap,
            sample_rate,
            drop_last,
            data_ty=data_type,
            channels_order=channels_order,
            sample_indexs=sample_indexs,
        )


class PILImageSequentialDataset(ImagesSequentialDataset):
    def __getitem__(self, index: int) -> Image.Image:
        data, sample_indexs = super().__getitem__(index)
        data = [Image.open(x) for x in data]
        return VideoDatasetOutput(data, sample_indexs)


class MoviepyVideoDataset(SequentialDataset):
    def __init__(
        self,
        path,
        time_size: int,
        step: int,
        overlap: int = None,
        sample_rate: int = 1,
        drop_last: bool = False,
        data_type: Literal["bgr", "rgb"] = "bgr",
        contenct_box: List[int] = None,
        sample_indexs: List[List[int]] = None,
    ) -> None:
        self.path = path
        self.f = self.prepare_raw_datas(self.path)
        super().__init__(
            self.f,
            time_size,
            step,
            overlap,
            sample_rate,
            drop_last,
            data_type=data_type,
            sample_indexs=sample_indexs,
        )
        self.contenct_box = contenct_box

    def prepare_raw_datas(self, path):
        f = VideoFileClip(path)
        return f

    def get_raw_data(self, index: int):
        return self.f.get_frame(index * 1 / self.f.fps)

    @property
    def fps(self):
        return self.f.fps

    @property
    def size(self):
        return self.f.size

    @property
    def total_frames(self):
        return int(self.duration * self.fps)

    @property
    def duration(self):
        return self.f.duration

    @property
    def width(self):
        return self.f.w

    @property
    def height(self):
        return self.f.h

    def __next__(
        self,
    ):
        video_clips = []
        cnt = 0
        frame_indexs = []

        for frame in itertools.islice(self.video.iter_frames(), step=self.step):
            if cnt >= self.total_frames:
                raise StopIteration
            else:
                frame_indexs.append(cnt)
                cnt += self.step
            if len(video_clips) < self.time_size:
                video_clips.append(frame)
            else:
                return_video_clips = video_clips
                return_frame_indexs = frame_indexs
                video_clips = []
                frame_indexs = []
                return VideoDatasetOutput(return_video_clips, return_frame_indexs)


class TorchVideoDataset(object):
    pass


class OpenCVVideoDataset(SequentialDataset):
    def __init__(
        self,
        path,
        time_size: int,
        step: int,
        overlap: int = None,
        sample_rate: int = 1,
        drop_last: bool = False,
        data_type: Literal["bgr", "rgb"] = "bgr",
        channels_order: str = "t h w c",
        sample_indexs: List[List[int]] = None,
    ) -> None:
        self.path = path
        self.f = self.prepare_raw_datas(path)
        super().__init__(
            self.f,
            time_size,
            step,
            overlap,
            sample_rate,
            drop_last,
            data_type=data_type,
            channels_order=channels_order,
            sample_indexs=sample_indexs,
        )

    def prepare_raw_datas(self, path):
        f = cv2.VideoCapture(path)
        return f

    def get_raw_data(self, index: int):
        self.f.set(cv2.CAP_PROP_POS_FRAMES, index)
        if index < 0 or index >= self.total_frames:
            raise IndexError(
                f"index must in [0, {self.total_frames -1 }], but given index"
            )
        ret, frame = self.f.read()
        if self.data_type == "rgb":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def get_raw_data_by_time(self, idx):
        raise NotImplementedError

    @property
    def total_frames(self):
        return int(self.f.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def width(self):
        return int(self.f.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self):
        return int(self.f.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def durtion(self):
        return self.total_frames / self.fps

    @property
    def fps(self):
        return self.f.get(cv2.CAP_PROP_FPS)


class DecordVideoDataset(SequentialDataset):
    def __init__(
        self,
        path,
        time_size: int,
        step: int,
        device: str,
        overlap: int = None,
        sample_rate: int = 1,
        drop_last: bool = False,
        device_id: int = 0,
        data_type: Literal["bgr", "rgb"] = "bgr",
        channels_order: str = "t h w c",
        sample_indexs: List[List[int]] = None,
    ) -> None:
        self.path = path
        self.device = device
        self.device_id = device_id
        self.f = self.prepare_raw_datas(path)
        super().__init__(
            self.f,
            time_size,
            step,
            overlap,
            sample_rate,
            drop_last,
            data_type=data_type,
            channels_order=channels_order,
            sample_indexs=sample_indexs,
        )

    def prepare_raw_datas(self, path):
        from decord import VideoReader
        from decord import cpu, gpu

        if self.device == "cpu":
            device = cpu(self.device_id)
        else:
            device = gpu(self.device_id)
        with open(path, "rb") as f:
            f = VideoReader(f, ctx=device)
        return f

    # decord 的 颜色通道 通道默认是 rgb
    def get_raw_data(self, index: int):
        data = self.f[index].asnumpy()
        if self.data_type == "bgr":
            data = data[:, :, ::-1]
        return data

    def get_batch_raw_data(self, indexs: List[int]):
        data = self.f.get_batch(indexs).asnumpy()

        if self.data_type == "bgr":
            data = data[:, :, :, ::-1]
        return data

    @property
    def total_frames(self):
        return len(self.f)

    @property
    def height(self):
        return self.f[0].shape[0]

    @property
    def width(self):
        return self.f[0].shape[1]

    @property
    def size(self):
        return self.f[0].shape[:2]

    @property
    def shape(self):
        return self.f[0].shape


class VideoMapClipDataset(SequentialDataset):
    def __init__(
        self,
        video_map: str,
        raw_datas,
        time_size: int,
        step: int,
        overlap: int = None,
        sample_rate: int = 1,
        drop_last: bool = False,
        max_num_per_batch: int = None,
    ) -> None:
        self.video_map = video_map
        super().__init__(
            raw_datas,
            time_size,
            step,
            overlap,
            sample_rate,
            drop_last,
            max_num_per_batch,
        )

    def generate_sample_idxs(self):
        # use video_map to generate matched sampled_index
        raise NotImplementedError
