import time


def get_current_strtime(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """get_current_strtime

    Args:
        fmt (_type_, optional): str time format. Defaults to "%Y-%m-%d %H:%M:%S".

    Returns:
        str: timestr
    """
    current_time = time.strftime(fmt, time.localtime())
    return current_time


def timestr_2_seconds(timestr: str) -> float:
    """convert timestr to time float num,

    Args:
        timestr (str): should be h:m:s or h:m:s:M or h:m:s.M

    Returns:
        float: seconds
    """
    timestr_lst = timestr.split(":")
    if len(timestr_lst) == 1:
        seconds = float(timestr_lst[0])
    else:
        if len(timestr_lst) == 3:
            time_range = [3600, 60, 1]
        elif len(timestr_lst) == 4:
            time_range = [3600, 60, 1, 1e-3]
            timestr_lst[-1] = timestr_lst[-1][:3]
        else:
            raise ValueError("timestr should be like h:m:s or h:m:s:M or h:m:s.M, but given {}".format(timestr))
        seconds = sum([float(timestr_lst[i]) * time_range[i] for i in range(len(timestr_lst))])
    return round(seconds, 3)
