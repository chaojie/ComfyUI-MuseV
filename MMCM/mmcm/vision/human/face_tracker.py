# limit the number of cpus used by high performance libraries
import os
from typing import Dict

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys
import argparse
import os
from pathlib import Path
import json
import traceback

import numpy as np
import torch


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def detect(tracker, device, source_path, video_map, opt):
    (
        out,
        show_vid,
        save_vid,
        save_txt,
        imgsz,
        evaluate,
        half,
        project,
        exist_ok,
        update,
        save_crop,
    ) = (
        opt.output,
        opt.show_vid,
        opt.save_vid,
        opt.save_txt,
        opt.imgsz,
        opt.evaluate,
        opt.half,
        opt.project,
        opt.exist_ok,
        opt.update,
        opt.save_crop,
    )
    from yolov5.utils.general import xyxy2xywh
    from yolov5.utils.torch_utils import select_device
    # Initialize
    device = select_device(device)
    half &= device.type != "cpu"  # half precision only supported on CUDA
    # initialize deepsort

    try:
        transition_data = video_map["clips"]
    except:
        print("no transition_data")
        transition_data = None
    try:
        c_box = video_map["content_box"]
    except:
        print("no content_box")
        c_box = None

    video_detect = json.load(open(source_path, encoding="UTF-8"))
    face_detections = video_detect["face_detections"]

    slice_id = 0
    for detects in face_detections:
        frame_idx = detects["frame_idx"]

        while (
            transition_data
            and (slice_id < len(transition_data))
            and (frame_idx >= transition_data[slice_id]["frame_end"])
        ):
            # print(frame_idx, transition_data[slice_id]['frame_end'])
            tracker.tracker.tracks = []
            slice_id += 1

        pred = detects["faces"]
        if pred is not None and len(pred):
            # Rescale boxes from img_size to im0 size
            det = []
            confs = []
            clss = []
            features = []
            for p in pred:
                det.append(p["bbox"])
                confs.append(float(p["det_score"]))
                features.append(p["embedding"])
                clss.append(0)
            det = np.array(det)
            confs = np.array(confs)
            clss = np.array(clss)
            features = torch.Tensor(features)

            xywhs = xyxy2xywh(det)

            # pass detections to deepsort
            if c_box:
                im0 = np.zeros((c_box[3] - c_box[1], c_box[2] - c_box[0]))
            else:
                im0 = np.zeros((video_map["height"], video_map["width"]))
            outputs = tracker.update(
                xywhs, confs, clss, im0, use_yolo_preds=True, features=features
            )

            assert len(pred) == len(outputs)
            for j, output in enumerate(outputs):
                bboxes = output[0:4]
                id = output[4]

                min_box_distance = np.inf
                match_p = None
                for p in pred:
                    if "trackid" not in p:
                        c_box_distance = abs(
                            bboxes - np.array(p["bbox"], dtype=np.int)
                        ).sum()
                        if c_box_distance < 10 and c_box_distance < min_box_distance:
                            match_p = p
                            min_box_distance = c_box_distance
                if match_p:
                    match_p["trackid"] = str(id)
                else:
                    print("not match: ", frame_idx, bboxes)
                    for p in pred:
                        print(p["bbox"])

        else:
            tracker.increment_ages()
        return video_map


class FaceTrackerByYolo5DeepSort(object):
    def __init__(
        self,
        config_file,
        device,
        deep_sort_model="osnet_ibn_x1_0_MSMT17",
        half: bool=False,
        
    ) -> None:
        from deep_sort.utils.parser import get_config
        from deep_sort.deep_sort import DeepSort
        cfg = get_config()
        cfg.merge_from_file(config_file)
        # Create as tracker
        self.tracker = DeepSort(
            deep_sort_model,
            device,
            max_dist=cfg.DEEPSORT.MAX_DIST,
            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
            max_age=cfg.DEEPSORT.MAX_AGE,
            n_init=cfg.DEEPSORT.N_INIT,
            nn_budget=cfg.DEEPSORT.NN_BUDGET,
        )

    def __call__(self, args, video_path, video_map, **kwds) -> Dict:
        """_summary_

        Args:
            args (_type_): _description_
            video_path (_type_): _description_
            save_path (_type_): _description_
            map_path (_type_): _description_
            kwds:
                # parser.add_argument('--yolo_model', nargs='+', type=str, default='yolov5m.pt', help='model.pt path(s)')
                parser.add_argument('--deep_sort_model', type=str, default='osnet_ibn_x1_0_MSMT17')
                # parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
                    '--output', type=str, default='inference/output', help='output folder'
                )  # output folder
                    '--imgsz',
                    '--img',
                    '--img-size',
                    nargs='+',
                    type=int,
                    default=[640],
                    help='inference size h,w',)
                    '--conf-thres', type=float, default=0.5, help='object confidence threshold')
                    '--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
                    '--fourcc',type=str,default='mp4v',
                    help='output video codec (verify ffmpeg support)',)
                    '--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
                    '--show-vid', action='store_true', help='display tracking video results')
                    '--save-vid', action='store_true', help='save video tracking results')
                    '--save-txt', action='store_true', help='save MOT compliant results to *.txt')
                # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
                    '--classes',
                    nargs='+',
                    type=int,
                    help='filter by class: --class 0, or --class 16 17',
                )
                    '--agnostic-nms', action='store_true', help='class-agnostic NMS'
                )
                parser.add_argument('--augment', action='store_true', help='augmented inference')
                parser.add_argument('--update', action='store_true', help='update all models')
                parser.add_argument('--evaluate', action='store_true', help='augmented inference')
                parser.add_argument(
                    "--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml"
                )
                    "--half", action="store_true", help="use FP16 half-precision inference"
                )
                parser.add_argument('--visualize', action='store_true', help='visualize features')
                    '--max-det', type=int, default=1000, help='maximum detection per image'
                )
                    '--save-crop', action='store_true', help='save cropped prediction boxes'
                )
                    '--dnn', action='store_true', help='use OpenCV DNN for ONNX inference'
                )
                    '--project', default=ROOT / 'runs/track', help='save results to project/name'
                )
                parser.add_argument('--name', default='exp', help='save results to project/name')
                    '--exist-ok', action='store_true',
                    help='existing project/name ok, do not increment',
                )

                    '-src_path',
                    type=str,
                    default='/innovation_cfs/entertainment/VideoMashup/video_face_moviepy/10fps',
                )
                    '-map_path', type=str,
                    default='/innovation_cfs/entertainment/VideoMashup/video_map/transnetv2_duration_frameidx_moviepy',

                    '-overwrite', default=False, action="store_true"
                )  # whether overwrite the existing results


        Returns:
            Dict: _description_
        """
        video_info = detect(args, self.tracker, video_path, video_map, **kwds)
        return video_info
