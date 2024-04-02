# -*- coding: utf8 -*-
from typing import Any


def load_rainbow_config(
    app_id: str, user_id: str, secret_key: str, group: str, env_name: str = "Default"
) -> Any:
    from rainbow_sdk.rainbow_client import RainbowClient

    init_param = {
        "connectStr": "api.rainbow.oa.com:8080",
        "isUsingFileCache": False,
        "fileCachePath": "/data/rainbow/",
        "tokenConfig": {"app_id": app_id, "user_id": user_id, "secret_key": secret_key},
    }
    rc = RainbowClient(init_param)
    res = rc.get_configs_v3(group, env_name=env_name)
    return res["data"]
