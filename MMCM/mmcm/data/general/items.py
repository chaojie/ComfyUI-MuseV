from collections import UserList
from collections.abc import Iterable
from typing import Iterator, Any, List

from ...utils.util import convert_class_attr_to_dict

__all__ = ["Item", "Items"]


class Item(object):
    def __init__(self) -> None:
        pass

    def to_dct(self, target_keys: List[str] = None, ignored_keys: List[str] = None):
        base_ignored_keys = [
            "kwargs",
        ]
        if isinstance(ignored_keys, list):
            base_ignored_keys.extend(ignored_keys)
        elif isinstance(ignored_keys, str):
            base_ignored_keys.append(ignored_keys)
        else:
            pass
        return convert_class_attr_to_dict(
            self, target_keys=target_keys, ignored_keys=base_ignored_keys
        )

    def preprocess(self):
        pass


class Items(UserList):
    def __init__(
        self,
        data: Any = None,
    ):
        if data is None:
            data = list()
        if not isinstance(data, list):
            data = [data]
        super().__init__(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def __delitem__(self, i):
        del self.data[i]

    def __setitem__(self, i, v):
        self.data[i] = v

    def insert(self, i, v):
        self.data.insert(i, v)

    def __str__(self):
        return str(self.data)

    def to_dct(self, target_keys: List[str] = None, ignored_keys: List[str] = None):
        items = [item.to_dct(target_keys, ignored_keys) for item in self.data]
        return items

    def __iter__(self) -> Iterator:
        return iter(self.data)

    def preprocess(self):
        pass
