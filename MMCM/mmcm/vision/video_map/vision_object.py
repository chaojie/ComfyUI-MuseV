from __future__ import annotations
from typing import Iterable, Union, List, Dict, Any
import numpy as np

from ...data import Items, Item
from ...utils.util import convert_class_attr_to_dict


class Object(Item):
    def __init__(
        self,
        bbox: list,
        category: str,
        det_score: float = None,
        kps: dict = None,
        name: str = None,
        text: str = None,
        text_type: str = None,
        obj_id: int = None,
        attributes: Dict = None,
        trackid: int = None,
        **kwargs,
    ) -> None:
        """_summary_

        Args:
            bbox (list): _description_
            category (str):  物体类别，动物、文本、人、人脸等可以检测出框的物体
            det_score (float, optional): _description_. Defaults to None.
            kps (dict, optional): _description_. Defaults to None.
            name (str, optional): 物体姓名. Defaults to None.
            text (str, optional): 可以用于OCR类的检测输出，具体描述该物体，可以是文本内容、类别具体描述、caption等. Defaults to None.
            text_type (str, optional): 字幕，水印等. Defaults to None.
        """
        self.bbox = bbox
        self.category = category
        self.det_score = det_score
        self.kps = kps
        self.name = name
        self.text = text
        self.text_type = text_type
        self.obj_id = obj_id
        self.attributes = attributes
        self.trackid = trackid
        self.__dict__.update(**kwargs)

    @property
    def width(self):
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self):
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self):
        return self.width * self.height


class Human(Object):
    pass


class OpticalCharacter(Object):
    pass


class Objects(Items):
    def __init__(self, datas: List[Object] = None):
        super().__init__(datas)
        self.objs = self.data

    def get_target_category(self, target_category: list) -> list:
        if not isinstance(target_category, list):
            target_category = [target_category]
        objs = Objects([obj for obj in self.objs if obj.category in target_category])
        return objs

    def get_max_bbox_obj(self):
        areas = [obj.area for obj in self.objs]
        max_index = np.argmax(areas)
        obj = self.objs[max_index]
        return obj

    def __len__(
        self,
    ):
        return len(self.objs)

    def __getitem__(self, i):
        """支持索引和切片操作，如果输入是整数则返回 Object ，如果是切片，则返回 Objects

        Args:
            i (int or slice): 索引

        Raises:
            ValueError: 需要按照给的输入类型索引

        Returns:
            Object or Objects:
        """
        if "int" in str(type(i)):
            i = int(i)
        if isinstance(i, int):
            obj = self.objs[i]
            return obj
        elif isinstance(i, Iterable):
            objs = [self.__getitem__(x) for x in i]
            objs = Objects(objs)
            return objs
        elif isinstance(i, slice):
            if i.step is None:
                step = 1
            else:
                step = i.step
            objs = [self.__getitem__(x) for x in range(i.start, i.stop, step)]
            objs = Objects(objs)
            return objs
        else:
            raise ValueError(
                "unsupported input, should be int or slice, but given {}, type={}".format(
                    i, type(i)
                )
            )


class Role(Item):
    def __init__(
        self,
        name: str = None,
        age: int = None,
        gender: str = None,
        gender_confidence: float = None,
        appearance_frequency: float = None,
        roleid: int = None,
        faceid: int = None,
        **kwargs,
    ) -> None:
        self.name = name
        self.age = age
        self.gender = gender
        self.gender_confidence = gender_confidence
        self.appearance_frequency = appearance_frequency
        self.roleid = roleid
        self.faceid = faceid
        self.__dict__.update(**kwargs)

    @classmethod
    def from_data(cls, data, **kwargs):
        return Role(**data, **kwargs)


class Roles(Items):
    def __init__(self, data: List[Role] = None, **kwargs):
        super().__init__(data)
        self.roles = self.data
        self.__dict__.update(**kwargs)

    @classmethod
    def from_data(cls, datas: List, role_kwargs=None, **kwargs) -> Roles:
        if role_kwargs is None:
            role_kwargs = {}
        roles = [Role.from_data(role, **role_kwargs) for role in datas]
        return Roles(roles, **kwargs)
