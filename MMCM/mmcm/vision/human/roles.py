import os
import json


def load_roles(path):
    face_name_dataset = {}
    for face_name in os.listdir(path):
        for face_result_file in os.listdir(os.path.join(path, face_name)):
            try:
                face_result = json.load(
                    open(
                        os.path.join(path, face_name, face_result_file),
                        encoding='UTF-8',
                    )
                )["face_detections"]
                if face_name in face_name_dataset:
                    face_name_dataset[face_name].append(face_result[0]["embedding"])
                else:
                    face_name_dataset[face_name] = [face_result[0]["embedding"]]
            except:
                print(face_name, face_result_file, "is wrong")
    return face_name_dataset
