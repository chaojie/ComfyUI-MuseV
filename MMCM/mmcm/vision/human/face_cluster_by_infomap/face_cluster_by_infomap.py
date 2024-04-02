# -*- coding: UTF-8 -*-
from typing import Dict
import time
from multiprocessing.dummy import Pool as Threadpool
from multiprocessing import Pool
import multiprocessing as mp
import os
import json
from collections import Counter
import argparse
import traceback
import random

from tqdm import tqdm
import numpy as np

from .utils import Timer
from .evaluation import evaluate, accuracy
from ..roles import load_roles

def l2norm(vec):
    """
    归一化
    :param vec:
    :return:
    """
    vec /= np.linalg.norm(vec, axis=1).reshape(-1, 1)
    return vec


def cosine_distance(a, b, data_is_normalized=False):
    """Compute pair-wise cosine distance between points in `a` and `b`.
    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    data_is_normalized : Optional[bool]
        If True, assumes rows in a and b are unit length vectors.
        Otherwise, a and b are explicitly normalized to lenght 1.
    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.
    """
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1.0 - np.dot(a, b.T)


def intdict2ndarray(d, default_val=-1):
    arr = np.zeros(len(d)) + default_val
    for k, v in d.items():
        arr[k] = v
    return arr


def read_meta(fn_meta, start_pos=0, verbose=True):
    """
    idx2lb：每一个顶点对应一个类
    lb2idxs：每个类对应一个id
    """
    lb2idxs = {}
    idx2lb = {}
    with open(fn_meta) as f:
        for idx, x in enumerate(f.readlines()[start_pos:]):
            lb = int(x.strip())
            if lb not in lb2idxs:
                lb2idxs[lb] = []
            lb2idxs[lb] += [idx]
            idx2lb[idx] = lb

    inst_num = len(idx2lb)
    cls_num = len(lb2idxs)
    if verbose:
        print("[{}] #cls: {}, #inst: {}".format(fn_meta, cls_num, inst_num))
    return lb2idxs, idx2lb


class knn:
    def __init__(self, feats, k, index_path="", verbose=True):
        pass

    def filter_by_th(self, i):
        th_nbrs = []
        th_dists = []
        nbrs, dists = self.knns[i]
        for n, dist in zip(nbrs, dists):
            if 1 - dist < self.th:
                continue
            th_nbrs.append(n)
            th_dists.append(dist)
        th_nbrs = np.array(th_nbrs)
        th_dists = np.array(th_dists)
        return th_nbrs, th_dists

    def get_knns(self, th=None):
        if th is None or th <= 0.0:
            return self.knns
        # TODO: optimize the filtering process by numpy
        # nproc = mp.cpu_count()
        nproc = 1
        with Timer("filter edges by th {} (CPU={})".format(th, nproc), self.verbose):
            self.th = th
            self.th_knns = []
            tot = len(self.knns)
            if nproc > 1:
                pool = mp.Pool(nproc)
                th_knns = list(
                    tqdm(pool.imap(self.filter_by_th, range(tot)), total=tot)
                )
                pool.close()
            else:
                th_knns = [self.filter_by_th(i) for i in range(tot)]
            return th_knns


class knn_faiss(knn):
    """
    内积暴力循环
    归一化特征的内积等价于余弦相似度
    """

    def __init__(self, feats, k, index_path="", knn_method="faiss-cpu", verbose=True):
        import faiss

        with Timer("[{}] build index {}".format(knn_method, k), verbose):
            knn_ofn = index_path + ".npz"
            if os.path.exists(knn_ofn):
                print("[{}] read knns from {}".format(knn_method, knn_ofn))
                self.knns = np.load(knn_ofn)["data"]
            else:
                feats = feats.astype("float32")
                size, dim = feats.shape
                if knn_method == "faiss-gpu":
                    import math

                    i = math.ceil(size / 1000000)
                    if i > 1:
                        i = (i - 1) * 4
                    res = faiss.StandardGpuResources()
                    res.setTempMemory(i * 1024 * 1024 * 1024)
                    index = faiss.GpuIndexFlatIP(res, dim)
                else:
                    index = faiss.IndexFlatIP(dim)
                index.add(feats)
        with Timer("[{}] query topk {}".format(knn_method, k), verbose):
            knn_ofn = index_path + ".npz"
            if os.path.exists(knn_ofn):
                pass
            else:
                sims, nbrs = index.search(feats, k=k)
                # torch.cuda.empty_cache()
                self.knns = [
                    (np.array(nbr, dtype=np.int32), 1 - np.array(sim, dtype=np.float32))
                    for nbr, sim in zip(nbrs, sims)
                ]


def knns2ordered_nbrs(knns, sort=True):
    if isinstance(knns, list):
        knns = np.array(knns)
    nbrs = knns[:, 0, :].astype(np.int32)
    dists = knns[:, 1, :]
    if sort:
        # sort dists from low to high
        nb_idx = np.argsort(dists, axis=1)
        idxs = np.arange(nb_idx.shape[0]).reshape(-1, 1)
        dists = dists[idxs, nb_idx]
        nbrs = nbrs[idxs, nb_idx]
    return dists, nbrs


# 构造边
def get_links(single, links, nbrs, dists, frame_idx, trackids, det_scores):
    for i in tqdm(range(nbrs.shape[0])):
        count = 0
        for j in range(0, len(nbrs[i])):
            # 排除本身节点
            if i == nbrs[i][j]:
                continue
            elif dists[i][j] > 1 - min_sim:
                break
            elif (nbrs[i][j], i) in links.keys():
                count += 1
                continue
            elif frame_idx[i][0] == frame_idx[nbrs[i][j]][0]:
                # links[(i, nbrs[i][j])] = -np.inf
                # count += 1
                continue
            elif det_scores[i] * det_scores[nbrs[i][j]] < 0.55:
                continue
            else:
                count += 1
                links[(i, nbrs[i][j])] = float(1 - dists[i][j])
                # links[(i, nbrs[i][j])] = det_scores[i] * det_scores[nbrs[i][j]] * float(1 - dists[i][j])
                # links[(i, nbrs[i][j])] = (1 if genders[i]==genders[nbrs[i][j]] else 0.8)*det_scores[i]*det_scores[nbrs[i][j]]*float(1 - dists[i][j])

        track_weight = 10
        if trackids[i] != "-1":
            trackids = np.array(trackids, dtype=np.int)
            same_trackid_idx = np.where(trackids == trackids[i])[0]
            for j in same_trackid_idx:
                if i != j:
                    if (j, i) in links.keys():
                        links[(j, i)] = track_weight
                    elif (i, j) in links.keys():
                        links[(i, j)] = track_weight
                    else:
                        links[(i, j)] = track_weight
                        count += 1

        # 统计孤立点
        if count == 0:
            single.append(i)
    return single, links


# 构造边
def get_links_directed(single, links, nbrs, dists, frame_idx, trackids, det_scores):
    for i in tqdm(range(nbrs.shape[0])):
        count = 0
        for j in range(0, len(nbrs[i])):
            # 排除本身节点
            if i == nbrs[i][j]:
                continue
            elif dists[i][j] > 1 - min_sim:
                break
            elif frame_idx[i][0] == frame_idx[nbrs[i][j]][0]:
                # links[(i, nbrs[i][j])] = -np.inf
                # count += 1
                continue
            elif det_scores[i] * det_scores[nbrs[i][j]] < 0.55:
                continue
            else:
                count += 1
                links[(i, nbrs[i][j])] = float(1 - dists[i][j])
                # links[(i, nbrs[i][j])] = det_scores[i] * det_scores[nbrs[i][j]] * float(1 - dists[i][j])
                # links[(i, nbrs[i][j])] = (1 if genders[i]==genders[nbrs[i][j]] else 0.8)*det_scores[i]*det_scores[nbrs[i][j]]*float(1 - dists[i][j])

        track_weight = 2
        if trackids[i] != "-1":
            trackids = np.array(trackids, dtype=np.int)
            same_trackid_idx = np.where(trackids == trackids[i])[0]
            for j in same_trackid_idx:
                if i != j:
                    if (i, j) in links.keys():
                        links[(i, j)] = track_weight
                    else:
                        links[(i, j)] = track_weight
                        count += 1

        # 统计孤立点
        if count == 0:
            single.append(i)
    return single, links


def cluster_by_infomap(
    nbrs, dists, pred_label_path, frame_idx, trackids, det_scores, save_result=False
):
    """
    基于infomap的聚类
    :param nbrs:
    :param dists:
    :param pred_label_path:
    :return:
    """
    import infomap
    single = []
    links = {}
    with Timer("get links", verbose=True):
        single, links = get_links_directed(
            single=single,
            links=links,
            nbrs=nbrs,
            dists=dists,
            frame_idx=frame_idx,
            det_scores=det_scores,
            trackids=trackids,
        )
    print("pair数量：{}".format(len(links.keys())))

    infomapWrapper = infomap.Infomap("--two-level --directed")
    for (i, j), sim in tqdm(links.items()):
        _ = infomapWrapper.addLink(int(i), int(j), sim)

    # 聚类运算
    infomapWrapper.run()

    label2idx = {}
    idx2label = {}

    # 聚类结果统计
    for node in infomapWrapper.iterTree():
        # node.physicalId 特征向量的编号
        # node.moduleIndex() 聚类的编号
        idx2label[node.physicalId] = node.moduleIndex()
        if node.moduleIndex() not in label2idx:
            label2idx[node.moduleIndex()] = []
        label2idx[node.moduleIndex()].append(node.physicalId)

    node_count = 0
    for k, v in label2idx.items():
        if k == 0:
            node_count += len(v[2:])
            label2idx[k] = v[2:]
            # print(k, v[2:])
        else:
            node_count += len(v[1:])
            label2idx[k] = v[1:]
            # print(k, v[1:])

    for i in range(nbrs.shape[0]):
        if (i not in idx2label.keys()) and (i not in single):
            single.append(i)
    # 孤立点个数
    print("孤立点数：{}".format(len(single)))

    keys_len = len(list(label2idx.keys()))
    # print(keys_len)

    # 孤立点放入到结果中
    for single_node in single:
        if single_node not in idx2label.keys():
            idx2label[single_node] = keys_len
            label2idx[keys_len] = [single_node]
            keys_len += 1

    print("总类别数：{}".format(keys_len))

    idx_len = len(list(idx2label.keys()))
    print("总节点数：{}".format(idx_len))

    # 保存结果
    if save_result:
        with open(pred_label_path, "w") as of:
            for idx in range(idx_len):
                of.write(str(idx2label[idx]) + "\n")

    if label_path is not None:
        pred_labels = intdict2ndarray(idx2label)
        true_lb2idxs, true_idx2lb = read_meta(label_path)
        gt_labels = intdict2ndarray(true_idx2lb)
        for metric in metrics:
            evaluate(gt_labels, pred_labels, metric)

    return idx2label, label2idx


def get_dist_nbr(features, k=80, knn_method="faiss-cpu"):
    features = l2norm(features)

    index = knn_faiss(feats=features, k=k, knn_method=knn_method)
    knns = index.get_knns()
    dists, nbrs = knns2ordered_nbrs(knns)
    return dists, nbrs


def detection_dic_to_list(face_detections):
    features = []
    frame_idx = []
    trackids = []
    det_scores = []
    genders = []
    track2idx = {}
    for i in range(len(face_detections)):
        if face_detections[i]["faces"]:
            for j in range(len(face_detections[i]["faces"])):
                features.append(face_detections[i]["faces"][j]["embedding"])
                frame_idx.append([i, j])
                det_scores.append(float(face_detections[i]["faces"][j]["det_score"]))
                genders.append(face_detections[i]["faces"][j]["gender"])
                try:
                    c_trackid = int(face_detections[i]["faces"][j]["trackid"])
                except:
                    c_trackid = -1
                    print(
                        "no trackid",
                        face_detections[i]["frame_idx"],
                        face_detections[i]["faces"][j]["bbox"],
                    )
                trackids.append(c_trackid)
                if c_trackid in track2idx.keys():
                    track2idx[c_trackid].append(len(features) - 1)
                else:
                    track2idx[c_trackid] = [len(features) - 1]

    return features, frame_idx, trackids, det_scores, genders, track2idx


def top_role(label2frame, idx2label, frameids, face_detections, max_id_num=20):
    """
    统计出镜率前max_id_num位的角色id
    :param label2frame:
    :param idx2label:
    :param max_id_num:
    :return:
    """
    len_list = list(map(len, list(map(set, list(label2frame.values())))))
    sort_idx = np.argsort(len_list)
    sort_idx = sort_idx[::-1]
    id_sort = np.array(list(label2frame.keys()))[sort_idx].tolist()
    top_id = id_sort[:max_id_num]
    print(top_id)
    top_id_info = {}
    for i in range(len(idx2label)):
        current_id = idx2label[i]
        if current_id in top_id:
            current_face = face_detections[frameids[i][0]]["faces"][frameids[i][1]]
            current_gender = float(current_face["gender"])
            current_det_score = float(current_face["det_score"])
            current_age = int(current_face["age"])
            current_frame = int(face_detections[frameids[i][0]]["frame_idx"])
            current_embedding = current_face["embedding"]
            if current_id in top_id_info.keys():
                top_id_info[current_id]["gender"].append(current_gender)
                top_id_info[current_id]["det_score"].append(current_det_score)
                top_id_info[current_id]["age"].append(current_age)
                top_id_info[current_id]["embedding"].append(current_embedding)
                top_id_info[current_id]["frame"].append(current_frame)
            else:
                top_id_info[current_id] = {
                    "gender": [current_gender],
                    "det_score": [current_det_score],
                    "age": [current_age],
                    "embedding": [current_embedding],
                    "frame": [current_frame],
                }

    leading_roles = []
    for i in top_id:
        current_gender_f = np.mean(top_id_info[i]["gender"])
        current_gender = "M" if current_gender_f > 0 else "F"
        current_age = int(np.round(np.mean(top_id_info[i]["age"])))
        info = {
            "name": "",
            "age": current_age,
            "gender": current_gender,
            "gender_confidence": round(current_gender_f, 3),
            "appearance_frequency": round(
                len(set(top_id_info[i]["frame"])) / len(face_detections), 3
            ),
            "roleid": top_id.index(i),
            "faceid": i,
            "embedding": random.sample(
                top_id_info[i]["embedding"], min(100, len(top_id_info[i]["embedding"]))
            ),
            "det_score_average": np.mean(top_id_info[i]["det_score"]),
            "det_score_max": np.max(top_id_info[i]["det_score"]),
        }
        leading_roles.append(info)
    return leading_roles, id_sort


def role2name(leading_roles, face_name_dataset, face_distance_threshold=0.75):
    for info in leading_roles:
        min_distance = np.inf
        name = ""
        for key, value in face_name_dataset.items():
            distance = cosine_distance(
                np.array(info["embedding"]), np.array(value)
            ).mean()
            if distance < min_distance and distance < face_distance_threshold:
                min_distance = distance
                name = key
        info["name"] = name
        if name:
            print(info["roleid"], name)
    return leading_roles


knn_method = "faiss-gpu"
metrics = ["pairwise", "bcubed", "nmi"]
min_sim = 0.4
k = 2048
# true_label
label_path = None


def predict(source_path, video_map, face_name_dataset, max_id_num=20):
    video_detect = video_map
    try:
        face_detections = video_detect["face_detections"]
    except:
        face_detections = video_detect["face_detections"]
        (
            features,
            frameids,
            trackids,
            det_scores,
            genders,
            track2idx,
        ) = detection_dic_to_list(face_detections)

        # 每个track挑选最多5个人脸进行聚类
        select_idx = []
        max_num_per_track = 5
        for key, value in track2idx.items():
            if key == -1 or len(value) <= max_num_per_track:
                select_idx.extend(value)
            else:
                duration = len(value) // max_num_per_track
                select_idx.extend(value[::duration])
        select_idx = list(set(select_idx))
        select_idx.sort()
        select_idx = np.array(select_idx)
        select_features = np.array(features, np.float32)[select_idx]
        select_trackids = np.array(trackids, dtype=np.int)[select_idx]
        select_det_scores = np.array(det_scores, np.float32)[select_idx]
        select_frameids = np.array(frameids, np.int)[select_idx]
        pred_label_path = "./part1_test_predict.txt"

        # 将挑选出来的人脸进行聚类
        with Timer("All face cluster step"):
            dists, nbrs = get_dist_nbr(
                features=select_features, k=k, knn_method=knn_method
            )
            print(dists.shape, nbrs.shape)
            part_idx2label, part_label2idx = cluster_by_infomap(
                nbrs,
                dists,
                pred_label_path,
                select_frameids,
                select_trackids,
                select_det_scores,
                save_result=False,
            )

        # 将聚类结果对应到所有人脸
        idx2label = {}
        label2idx = {}
        label2frame = {}
        for i in range(len(select_idx)):
            o_i = select_idx[i]
            label = part_idx2label[i]
            idx2label[o_i] = part_idx2label[i]
            if label in label2idx.keys():
                label2idx[label].append(o_i)
                label2frame[label].append(frameids[o_i][0])
            else:
                label2idx[label] = [o_i]
                label2frame[label] = [frameids[o_i][0]]
        for i in range(len(features)):
            if i not in idx2label.keys():
                c_labels = []
                c_trackid = trackids[i]
                same_trackid_idx = track2idx[c_trackid]
                same_trackid_idx.remove(i)
                for j in same_trackid_idx:
                    if (i != j) and (j in idx2label.keys()):
                        c_labels.append(idx2label[j])
                if len(c_labels) > 0:
                    label = Counter(c_labels).most_common()[0][0]
                else:
                    label = len(list(label2idx.keys()))
                idx2label[i] = label
                if label in label2idx.keys():
                    label2idx[label].append(i)
                    label2frame[label].append(frameids[i][0])
                else:
                    label2idx[label] = [i]
                    label2frame[label] = [frameids[i][0]]

        leading_roles, id_sort = top_role(
            label2frame, idx2label, frameids, face_detections, max_id_num=20
        )

        video_detect["role_info"] = {
            "role_num": len(label2frame),
            "leading_roles": leading_roles,
        }

        # 给每个detection分配faceid和roleid
        assert len(features) == len(list(idx2label.keys()))
        for i in range(len(features)):
            label = idx2label[i]
            face_detections[frameids[i][0]]["faces"][frameids[i][1]]["faceid"] = label
            face_detections[frameids[i][0]]["faces"][frameids[i][1]][
                "roleid"
            ] = id_sort.index(label)

    video_detect["role_info"]["leading_roles"] = role2name(
        video_detect["role_info"]["leading_roles"], face_name_dataset
    )

    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_path, "w", encoding="utf-8") as fp:
        json.dump(video_detect, fp, ensure_ascii=False, indent=4)

    # 将人脸聚类信息整合进video_map
    video_map = json.load(open(map_path, encoding="UTF-8"))
    if "single_frame_transiton_score" in video_map:
        del video_map["single_frame_transiton_score"]
    if "all_frame_transiton_score" in video_map:
        del video_map["all_frame_transiton_score"]
    video_map["role_info"] = video_detect["role_info"]
    video_map["face_num_per_frame"] = sum(
        [len(i["faces"]) if i["faces"] else 0 for i in face_detections]
    ) / len(face_detections)
    # video_map["role_num_per_frame"] = sum(len_list)  / len(face_detections)

    face_idx = 0
    truncate_time = 0.1
    truncate_frame = int(video_map["sample_fps"] * truncate_time)
    for slice in video_map["clips"]:
        frame_start = slice["frame_start"]
        frame_end = slice["frame_end"]
        if (slice["cliptype"] == "body") and (
            frame_end - frame_start + 1 > 2 * truncate_frame
        ):
            frame_start += truncate_frame
            frame_end -= truncate_frame
        frame_num = 0
        face_num = 0
        roles = {}

        while face_idx < len(face_detections) and (
            face_detections[face_idx]["frame_idx"] < frame_start
        ):
            face_idx += 1

        while face_idx < len(face_detections) and (
            frame_start <= face_detections[face_idx]["frame_idx"] <= frame_end
        ):
            if face_detections[face_idx]["faces"]:
                for face in face_detections[face_idx]["faces"]:
                    c_roleid = face["roleid"]
                    if (c_roleid is not None) and (c_roleid < 1000):
                        if c_roleid in roles:
                            if (
                                face_detections[face_idx]["frame_idx"]
                                in roles[c_roleid]["bbox"]
                            ):
                                roles[c_roleid]["bbox"][
                                    face_detections[face_idx]["frame_idx"]
                                ].append(face["bbox"])
                            else:
                                roles[c_roleid]["bbox"][
                                    face_detections[face_idx]["frame_idx"]
                                ] = [face["bbox"]]
                        else:
                            roles[c_roleid] = {
                                "bbox": {
                                    face_detections[face_idx]["frame_idx"]: [
                                        face["bbox"]
                                    ]
                                },
                                "name": video_detect["role_info"]["leading_roles"][
                                    c_roleid
                                ]["name"]
                                if c_roleid
                                < len(video_detect["role_info"]["leading_roles"])
                                else "",
                            }
                face_num += len(face_detections[face_idx]["faces"])
            face_idx += 1
            frame_num += 1
        slice["roles"] = roles
        slice["face_num_per_frame"] = (face_num / frame_num) if frame_num > 0 else None
        slice["role_num_per_frame"] = (
            (len(roles) / frame_num) if frame_num > 0 else None
        )
        if "feat" in slice:
            del slice["feat"]
    return video_map


class FaceClusterByInfomap(object):
    def __init__(
        self,
        roles_path,
    ):
        self.roles_dataset = load_roles(roles_path)

    def __call__(
        self,
        source_path,
        video_map,
    ):
        video_map = predict(
            source_path,
            video_map,
            self.roles_dataset,
            max_id_num=20,
        )
        return video_map
