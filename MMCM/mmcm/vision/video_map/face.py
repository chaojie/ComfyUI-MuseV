import os
from typing import Dict, List

from ...utils.path_util import get_file_name_ext
from ...utils.util import load_dct_from_file
from .vision_object import Object
from .vision_frame import Frame, FrameSeq


def face_meta_2_tme_meta(src: dict) -> dict:
    """人脸中的元信息格式转换

    Args:
        src (dict): 人脸中的元信息

    Returns:
        dict: 转换后的元信息
    """
    dst = {}
    dst["media_name"] = src["video_name"]
    dst["mediaid"] = src["video_name"]
    dst["signature"] = src["video_file_hash_code"]
    dst["fps"] = src["fps"]
    dst["duration"] = src["duration"]
    dst["frame_num"] = src["frame_num"]
    dst["height"] = src["height"]
    dst["width"] = src["width"]
    return dst


def face_obj_2_tme_obj(src: dict) -> dict:
    """人脸信息转换为 Object中的元信息

    Args:
        src (dict): 人脸框相关信息

    Returns:
        dict: 转换后的人脸信息
    """
    obj = {}
    obj["category"] = "face"
    obj["bbox"] = src["bbox"]
    obj["kps"] = src["kps"]
    obj["det_score"] = src["det_score"]
    obj["gender"] = src["gender"]
    obj["age"] = src["age"]
    obj["trackid"] = src["roleid"]
    return obj


def face_clips_2_tme_clips(src: list) -> list:
    """人脸信息转换为Clip

    Args:
        src (list): 人脸中 Clip 的多帧检测信息

    Returns:
        list: Clip 中的 frames信息
    """
    dst = []
    for idx, frame_perception in enumerate(src):
        frame_dst = {}
        frame_dst["frame_idx"] = frame_perception["frame_idx"]
        objs = []
        if frame_perception["faces"] is not None:
            for face in frame_perception["faces"]:
                obj = face_obj_2_tme_obj(face)
                objs.append(obj)
        frame_dst["objs"] = objs
        dst.append(frame_dst)
    return dst


def face2TMEType(src: dict) -> dict:
    """人脸检测的信息转换成 视频剪辑中的格式

    Args:
        src (dict): 人脸检测信息

    Returns:
        dict: 转换后的字典格式
    """
    meta_info = face_meta_2_tme_meta(
        {
            k: v
            for k, v in src.items()
            if k
            not in [
                "face_detections",
                "single_frame_transiton_score",
                "all_frame_transiton_score",
                "clips",
            ]
        }
    )
    clips = face_clips_2_tme_clips(src["face_detections"])
    video_info = {"meta_info": meta_info, "sub_meta_info": [], "clips": clips}
    return video_info


def load_multi_face(
    path_lst: str,
) -> dict:
    """读取多个人脸检测结果文件，转化成VideoInfo对应的字典格式。

    Args:
        path_lst (str or [str]): 人脸检测结果文件

    Returns:
        dict: VideoInfo对应的字典格式, key是 文件名
    """
    if not isinstance(path_lst, list):
        path_lst = [path_lst]
    face_info_dct = {}
    for path in path_lst:
        filename, ext = get_file_name_ext(os.path.basename(path))
        face_info = load_dct_from_file(path)
        face_info = face2TMEType(face_info)
        face_info_dct[filename] = face_info
    return face_info_dct


def face_roles2frames(src: dict, **kwargs: dict) -> List[Frame]:
    """将roles字典转换为Frame

    Args:
        src (dict): {
            roleid: {
                "bbox": {
                "frame_idx": [
                        [x1, y1, x2, y2]
                    ]
                }
                "names": str,
            }
        }
        kwargs (dict): 便于其他需要的参数也传到Frame中去

    Returns:
        List[Frame]: _description_
    """
    frames = {}
    for roleid, faces_info in src.items():
        if "name" not in faces_info or faces_info["name"] == "":
            name = "unknown"
        else:
            name = faces_info["name"]
        if "bbox" in faces_info:
            frames_bbox = faces_info["bbox"]
            for frameid, bbox in frames_bbox.items():
                frameid = int(frameid)
                if frameid not in frames:
                    frames[frameid] = {"objs": [], "frame_idx": frameid}
                obj = {
                    "name": name,
                    "bbox": bbox[0],
                    "category": "person",
                    "obj_id": int(roleid),
                }
                obj = Object(**obj)
                frames[frameid]["objs"].append(obj)
    frame_obj_list = []
    for frameid in sorted(frames.keys()):
        frame_args = frames[frameid]
        frame_args.update(**kwargs)
        frame = Frame(**frame_args)
        frame_obj_list.append(frame)
    return frame_obj_list


def clipseq_face_roles2frames(clips_roles: List[Dict], **kwargs: dict) -> FrameSeq:
    frame_seq = []
    for roles in clips_roles:
        frames = face_roles2frames(roles)
        frame_seq.extend(frames)
    frame_seq = sorted(frame_seq, key=lambda f: f.frame_idx)
    return FrameSeq(frame_seq, **kwargs)
