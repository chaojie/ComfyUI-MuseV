from typing import Callable

from ..data.video_dataset import SequentialDataset
# import torchvision.transforms.transforms import 


def inference_video(video_dataset: SequentialDataset, predictor: Callable, transform:Callable=None, post_process: Callable=None):
    results = []
    for data in video_dataset:
        if transform is not None:
            data = transform(data)
            data = predictor(data)
    if post_process is not None:
        results = post_process(results)
    return results
