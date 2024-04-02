from typing import Any, List

import logging

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Register:
    def __init__(self, registry_name: str):
        self._dict = {}
        self._name = registry_name

    def __setitem__(self, key: str, value: Any) -> None:
        if not callable(value):
            raise Exception(f"Value of a Registry must be a callable!\nValue: {value}")
        # 优先使用自定义的name，其次使用类名或者函数名。
        if "name" in value.__dict__:
            key = value.name
        elif key is None:
            key = value.__name__
        if key in self._dict:
            logger.warning("Key %s already in registry %s." % (key, self._name))
        self._dict[key] = value

    def register(self, target: str) -> Any:
        """Decorator to register a function or class."""

        def add(key, value):
            self[key] = value
            return value

        if callable(target):
            # @reg.register
            return add(None, target)
        # @reg.register('alias')
        return lambda x: add(target, x)

    def __getitem__(self, key: str) -> Any:
        return self._dict[key]

    def __contains__(self, key: str) -> bool:
        return key in self._dict

    def keys(self) -> List[str]:
        """key"""
        return self._dict.keys()
