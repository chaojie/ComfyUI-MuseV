from typing import Union, List, Dict, Tuple, Literal

import logging


def convert_byte_unit(
    value: float,
    src_unit: Literal["b", "B", "KB", "MB", "GB", "TB"],
    target_unit: Literal["b", "B", "KB", "MB", "GB", "TB"],
) -> float:
    """convert value in src_unit to target_unit. Firstlt, all src_unit to Byte, then to target_unit

    Args:
        value (float): _description_
        src_unit (Literal[&quot;b&quot;, &quot;B&quot;, &quot;KB&quot;, &quot;MB&quot;, &quot;GB&quot;, &quot;TB&quot;]): _description_
        target_unit (Literal[&quot;b&quot;, &quot;B&quot;, &quot;KB&quot;, &quot;MB&quot;, &quot;GB&quot;, &quot;TB&quot;]): _description_

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        float: _description_
    """
    if src_unit in ["b", "bit"]:
        value = value / 8
    elif src_unit in ["B", "Byte"]:
        pass
    elif src_unit == "KB":
        value = value * 1024
    elif src_unit == "MB":
        value = value * 1024**2
    elif src_unit == "GB":
        value = value * (1024**3)
    elif src_unit == "TB":
        value = value * (1024**4)
    else:
        raise ValueError("src_unit is not valid")
    if target_unit in ["b", "bit"]:
        target_value = value * 8
    elif target_unit in ["B", "Byte"]:
        target_value = value
    elif target_unit == "KB":
        target_value = value / 1024
    elif target_unit == "MB":
        target_value = value / 1024**2
    elif target_unit == "GB":
        target_value = value / (1024**3)
    elif target_unit == "TB":
        target_value = value / (1024**4)
    else:
        raise ValueError("target_unit is not valid")
    return target_value


def get_gpu_status(unit="MB") -> List[Dict]:
    import pynvml

    try:
        infos = []

        # 初始化 pynvml
        pynvml.nvmlInit()
        # 获取 GPU 数量
        deviceCount = pynvml.nvmlDeviceGetCount()

        # 获取每个 GPU 的信息
        for i in range(deviceCount):
            gpu_info = {}
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_name = pynvml.nvmlDeviceGetName(handle)
            gpu_info = {
                "gpu_name": gpu_name,
                "total_memory": convert_byte_unit(
                    info.total, src_unit="B", target_unit=unit
                ),
                "used_memory": convert_byte_unit(
                    info.used, src_unit="B", target_unit=unit
                ),
                "used_memory_ratio": info.used / info.total,
                "gpu_utilization": utilization.gpu,
                "free_memory_ratio": info.free / info.total,
                "free_memory": convert_byte_unit(
                    info.free, src_unit="B", target_unit=unit
                ),
            }
            infos.append(gpu_info)
        # 释放 pynvml
        pynvml.nvmlShutdown()
    except Exception as e:
        print("get_gpu_status failed")
        logging.exception(e)
    return infos
