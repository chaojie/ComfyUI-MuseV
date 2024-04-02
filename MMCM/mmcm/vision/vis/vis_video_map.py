# -*- encoding: utf-8 -*-
import sys
import json

import cv2
from moviepy.editor import VideoFileClip


def vis_video_map(video_path, video_map, save_path):
    from yolov5.utils.plots import Annotator, colors
    if isinstance(video_map, str):
        video_map = json.load(open(video_map, encoding="UTF-8"))
    face_detections = []
    for i in video_map["face_detections"]:
        if i["faces"] and len(i["faces"]) > 0:
            face_detections.append(i)
    video_path = video_map["video_path"]
    # Capture video
    video = VideoFileClip(video_path)
    video = video.crop(*video_map["content_box"])
    fps = video.fps
    duration = video.duration
    width, height = video.size
    print("fps, duration, width, height:", fps, duration, width, height)
    vid_writer = cv2.VideoWriter(
        save_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        video_map["detect_fps"],
        (width, height),
    )
    frame_idx = 0
    face_idx = 0
    for im in video.iter_frames(fps=video_map["sample_fps"]):
        if face_idx == len(face_detections):
            break
        if frame_idx == 50000:
            break
        if frame_idx == face_detections[face_idx]["frame_idx"]:
            print(frame_idx)
            pred = face_detections[face_idx]["faces"]
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            annotator = Annotator(im, line_width=2, pil=not ascii)
            if pred is not None and len(pred):
                for p in pred:
                    conf = float(p["det_score"])
                    bboxes = p["bbox"]
                    track_id = p["trackid"]
                    c_gender = float(p["gender"])
                    c_age = p["age"]
                    if "roleid" in p:
                        roleid = p["roleid"]
                        if roleid >= 20:
                            continue
                        role = video_map["role_info"]["leading_roles"][roleid]
                        role_gender = role["gender_confidence"]
                        role_age = role["age"]
                        label = f"{track_id} {c_gender:.3f} {c_age} {conf:.2f} {roleid} {role_gender} {role_age}"
                        # label = f'{track_id} {c_gender:.3f} {c_age} {conf:.2f} {roleid}'
                    else:
                        label = f"{track_id} {c_gender:.3f} {c_age} {conf:.2f}"
                    annotator.box_label(bboxes, label, color=colors(0, True))
            else:
                print("No detections")
            # Stream results
            im0 = annotator.result()
            vid_writer.write(im0)
            face_idx += 1
        frame_idx += 1
