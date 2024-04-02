from genericpath import isfile
import re
import os

from ...text.utils.read_text import read_xml2json


# 一个正则表达式非常好用的网站
# https://regex101.com/r/cW8jA6/2


CHINESE_PATTERN = r"[\u4e00-\u9fff]+"
NOT_CHINESE_PATTERN = r"[^\u4e00-\u9fa5]"
ENGLISH_CHARACHTER_PATTERN = r"[a-zA-Z]+"
WORD_PATTERN = r"\w+"  # equal to [a-zA-Z0-9_].
NOT_WORD_PATTERN = r"\W+"


def has_target_string(lyric: str, pattern: str) -> bool:
    """本句歌词是否有目标字符串

    Args:
        lyric (str):
        pattern (str): 目标字符串的正则表达式式patteren

    Returns:
        bool: 有没有目标字符串
    """
    matched = re.findall(pattern, lyric)
    flag = len(matched) > 0
    return flag


def has_chinese_char(lyric: str) -> bool:
    """是否有中文字符

    Args:
        lyric (str):

    Returns:
        bool: 是否有中文字符
    """
    return has_target_string(lyric, CHINESE_PATTERN)


def has_non_chinese_char(lyric: str) -> bool:
    """是否有非中文字符，参考https://git.woa.com/innovative_tech/CopyrightGroup/LyricTools/blob/master/lyric_tools/dataProcess.py#L53

    Args:
        lyric (str):

    Returns:
        bool: 是否有中文字符
    """
    return has_target_string(lyric, NOT_CHINESE_PATTERN)


def has_english_alphabet_char(lyric: str) -> bool:
    """是否有英文字母表字符

    Args:
        lyric (str):

    Returns:
        bool:
    """
    return has_target_string(lyric, ENGLISH_CHARACHTER_PATTERN)


def check_is_lyric_row(lyric: str) -> bool:
    """该字符串是否是歌词

    Args:
        lyric (str): 待判断的字符串

    Returns:
        bool: 该字符串是否是歌词
    """
    is_not_lyric = [
        re.search(r"\[ti[:：]?", lyric),
        re.search(r"\[ar[:：]?", lyric),
        re.search(r"\[al[:：]?", lyric),
        re.search(r"\[by[:：]?", lyric),
        re.search(r"\[offset[:：]?", lyric),
        re.search(r"词[:：]?\(\d+,\d+\)[:：]?", lyric),
        re.search(r"曲[:：]?\(\d+,\d+\)[:：]?", lyric),
        re.search(r"作\(\d+,\d+\)词[:：]?", lyric),
        re.search(r"作\(\d+,\d+\)曲[:：]?", lyric),
        re.search(r"演\(\d+,\d+\)唱[:：]?", lyric),
        re.search(r"编\(\d+,\d+\)曲[:：]?", lyric),
        re.search(r"吉\(\d+,\d+\)他[:：]", lyric),
        re.search(r"人\(\d+,\d+\)声\(\d+,\d+\)录\(\d+,\d+\)音\(\d+,\d+\)师[:：]?", lyric),
        re.search(r"人\(\d+,\d+\)声\(\d+,\d+\)录\(\d+,\d+\)音\(\d+,\d+\)棚[:：]?", lyric),
        re.search(r"Vocal\s+\(\d+,\d+\)edite[:：]?", lyric),
        re.search(r"混\(\d+,\d+\)音\(\d+,\d+\)/\(\d+,\d+\)母\(\d+,\d+\)带[:：]?", lyric),
        re.search(r"混\(\d+,\d+\)音", lyric),
        re.search(r"和\(\d+,\d+\)声\(\d+,\d+\)编\(\d+,\d+\)写[:：]?", lyric),
        re.search(
            r"词\(\d+,\d+\)版\(\d+,\d+\)权\(\d+,\d+\)管\(\d+,\d+\)理\(\d+,\d+\)方[:：]?", lyric
        ),
        re.search(
            r"曲\(\d+,\d+\)版\(\d+,\d+\)权\(\d+,\d+\)管\(\d+,\d+\)理\(\d+,\d+\)方[:：]?", lyric
        ),
        re.search(r"联\(\d+,\d+\)合\(\d+,\d+\)出\(\d+,\d+\)品[:：]?", lyric),
        re.search(r"录\(\d+,\d+\)音\(\d+,\d+\)作\(\d+,\d+\)品", lyric),
        re.search(
            r"录\(\d+,\d+\)音\(\d+,\d+\)作\(\d+,\d+\)品\(\d+,\d+\)监\(\d+,\d+\)制[:：]?", lyric
        ),
        re.search(r"制\(\d+,\d+\)作\(\d+,\d+\)人[:：]?", lyric),
        re.search(r"制\(\d+,\d+\)作\(\d+,\d+\)人[:：]?", lyric),
        re.search(r"不\(\d+,\d+\)得\(\d+,\d+\)翻\(\d+,\d+\)唱", lyric),
        re.search(r"未\(\d+,\d+\)经\(\d+,\d+\)许\(\d+,\d+\)可", lyric),
        re.search(r"酷\(\d+,\d+\)狗\(\d+,\d+\)音\(\d+,\d+\)乐", lyric),
        re.search(r"[:：]", lyric),
    ]
    is_not_lyric = [x is not None for x in is_not_lyric]
    is_not_lyric = any(is_not_lyric)
    is_lyric = not is_not_lyric
    return is_lyric


def lyric2clip(lyric: str) -> dict:
    """convert a line of lyric into a clip
    Clip定义可以参考 https://git.woa.com/innovative_tech/VideoMashup/blob/master/videomashup/media/clip.py
    Args:
        lyric (str): _description_

    Returns:
        dict: 转化成Clip 字典
    """
    time_str_groups = re.findall(r"\d+,\d+", lyric)
    line_time_start = round(int(time_str_groups[0].split(",")[0]) / 1000, 3)
    line_duration = round(int(time_str_groups[0].split(",")[-1]) / 1000, 3)
    line_end_time = line_time_start + line_duration
    last_word_time_start = round(int(time_str_groups[-1].split(",")[0]) / 1000, 3)
    last_word_duration = round(int(time_str_groups[-1].split(",")[-1]) / 1000, 3)
    last_word_end_time = last_word_time_start + last_word_duration
    actual_duration = min(line_end_time, last_word_end_time) - line_time_start
    lyric = re.sub(r"\[\d+,\d+\]", "", lyric)

    # by yuuhong: 把每个字的起始时间点、结束时间点、具体的字拆分出来
    words_with_timestamp = get_words_with_timestamp(lyric)

    lyric = re.sub(r"\(\d+,\d+\)", "", lyric)
    dct = {
        "time_start": line_time_start,
        "duration": actual_duration,
        "text": lyric,
        "original_text": lyric,
        "timepoint_type": -1,
        "clips": words_with_timestamp,
    }
    return dct


# by yuuhong
# 把一句QRC中的每个字拆分出来
# lyric示例：漫(17316,178)步(17494,174)走(17668,193)在(17861,183) (18044,0)莎(18044,153)玛(18197,159)丽(18356,176)丹(18532,200)
def get_words_with_timestamp(lyric):
    words_with_timestamp = []
    elements = lyric.split(")")
    for element in elements:
        sub_elements = element.split("(")
        if len(sub_elements) != 2:
            continue
        text = sub_elements[0]
        timestamp = sub_elements[1]
        if re.match(r"\d+,\d+", timestamp):
            # 有效时间戳
            time_start_str = timestamp.split(",")[0]
            time_start = round(int(time_start_str) / 1000, 3)
            duration_str = timestamp.split(",")[1]
            duration = round(int(duration_str) / 1000, 3)
            clip = {"text": text, "time_start": time_start, "duration": duration}
            words_with_timestamp.append(clip)
    return words_with_timestamp


def lyric2clips(lyric: str, th: float = 0.75) -> list:
    """将一句歌词转换为至少1个的clip。拆分主要是针对中文空格拆分，如果拆分后片段过短，也会整句处理。
    Args:
        lyric (str): such as [173247,3275]去(173247,403)吗(173649,677) 配(174326,189)吗(174516,593) 这(175108,279)
        th (float, optional): 后面如果拆分后片段过短，也会整句处理. Defaults to 1.0.

    Returns:
        list: 歌词Clip序列
    """
    # 目前只对中文的一句歌词按照空格拆分，如果是英文空格则整句处理
    # 后面如果拆分后片段过短，也会整句处理
    if has_english_alphabet_char(lyric):
        return [lyric2clip(lyric)]
    splited_lyric = lyric.split(" ")
    if len(splited_lyric) == 1:
        return [lyric2clip(splited_lyric[0])]
    line_time_str, sub_lyric = re.split(r"]", splited_lyric[0])
    line_time_groups = re.findall(r"\d+,\d+", line_time_str)
    line_time_start = round(int(line_time_groups[0].split(",")[0]) / 1000, 3)
    line_duration = round(int(line_time_groups[0].split(",")[-1]) / 1000, 3)
    splited_lyric[0] = sub_lyric
    # 歌词xml都是歌词仅跟着时间，如果有空格 空格也应该是在时间后面，但有时候空格却在字后面、在时间前，因此需要修正
    # 错误的：[173247,3275]去(173247,403)吗 (173649,677)配(174326,189)吗 (174516,593)这(175108,279)
    # 错误的：[46122,2082]以(46122,213)身(46335,260)淬(46595,209)炼(46804,268)天(47072,250)地(47322,370)造(47692,341)化 (48033,172)
    # 修正成：[173247,3275]去(173247,403)吗(173649,677) 配(174326,189)吗(174516,593) 这(175108,279)
    for i in range(len(splited_lyric)):
        if splited_lyric[i] == "":
            del splited_lyric[i]
            break
        if splited_lyric[i][-1] != ")":
            next_lyric_time_start = re.search(
                r"\(\d+,\d+\)", splited_lyric[i + 1]
            ).group(0)
            splited_lyric[i] += next_lyric_time_start
            splited_lyric[i + 1] = re.sub(
                next_lyric_time_start, "", splited_lyric[i + 1]
            )
            splited_lyric[i + 1] = re.sub("\(\)", "", splited_lyric[i + 1])
    lyric_text = re.sub(r"\[\d+,\d+\]", "", lyric)
    lyric_text = re.sub(r"\(\d+,\d+\)", "", lyric_text)
    clips = []
    has_short_clip = False
    for sub_lyric in splited_lyric:
        sub_lyric_groups = re.findall(r"\d+,\d+", sub_lyric)
        sub_lyric_1st_word_time_start = round(
            int(sub_lyric_groups[0].split(",")[0]) / 1000, 3
        )
        sub_lyric_last_word_time_start = round(
            int(sub_lyric_groups[-1].split(",")[0]) / 1000, 3
        )
        sub_lyric_last_word_duration = round(
            int(sub_lyric_groups[-1].split(",")[-1]) / 1000, 3
        )
        sub_lyric_last_word_time_end = (
            sub_lyric_last_word_time_start + sub_lyric_last_word_duration
        )
        sub_lyric_duration = (
            sub_lyric_last_word_time_end - sub_lyric_1st_word_time_start
        )
        if sub_lyric_duration <= th:
            has_short_clip = True
            break
        sub_lyric_text = re.sub(r"\[\d+,\d+\]", "", sub_lyric)
        sub_lyric_text = re.sub(r"\(\d+,\d+\)", "", sub_lyric_text)
        # 使用原始lyric，而不是sub_lyric_text 主要是保留相关clip的歌词信息，便于语义连续
        dct = {
            "time_start": sub_lyric_1st_word_time_start,
            "duration": sub_lyric_duration,
            "text": sub_lyric_text,
            "original_text": lyric_text,
            "timepoint_type": -1,
        }
        clips.append(dct)
    if has_short_clip:
        clips = [lyric2clip(lyric)]
    return clips


def is_songname(lyric: str) -> bool:
    """是否是歌名，歌名文本含有ti, 如[ti:霍元甲 (《霍元甲》电影主题曲)]

    Args:
        lyric (str):

    Returns:
        bool:
    """
    return has_target_string(lyric, r"\[ti[:：]?")


def get_songname(lyric: str) -> str:
    """获取文本中的歌名，输入必须类似[ti:霍元甲 (《霍元甲》电影主题曲)]

    Args:
        lyric (str): 含有歌名的QRC文本行

    Returns:
        str: 歌名
    """
    return lyric.split("(")[0][4:-1]


def is_album(lyric: str) -> bool:
    """是否含有专辑名，文本必须类似[al:霍元甲]

    Args:
        lyric (str): _description_

    Returns:
        bool: _description_
    """

    return has_target_string(lyric, r"\[al[:：]?")


def get_album(lyric: str) -> str:
    """提取专辑名，文本必须类似[al:霍元甲]


    Args:
        lyric (str): 含有专辑名的QRC文本行

    Returns:
        str: 专辑名
    """
    return lyric[4:-1]


def is_singer(lyric: str) -> bool:
    """是否有歌手名，目标文本类似 [ar:周杰伦]

    Args:
        lyric (str): _description_

    Returns:
        bool: _description_
    """
    return has_target_string(lyric, r"\[ar[:：]?")


def get_singer(lyric: str) -> str:
    """提取歌手信息，文本必须类似[ar:周杰伦]

    Args:
        lyric (str): 含有歌手名的QRC文本行

    Returns:
        str: 歌手名
    """
    return lyric[4:-1]


def lyric2musicinfo(lyric: str) -> dict:
    """convert lyric content from str into musicinfo, a dict
    参考https://git.woa.com/innovative_tech/VideoMashup/blob/master/videomashup/media/media_info.py#L19
    {
        "meta_info": {},
        "sub_meta_info": [],
        "clips": [
            clip
        ]
    }

    Args:
        lyric (str): 来自QRC的歌词字符串

    Returns:
        musicinfo: 音乐谱面字典，https://git.woa.com/innovative_tech/VideoMashup/blob/master/videomashup/media/media_info.py#L19
    """
    lyrics = lyric["QrcInfos"]["LyricInfo"]["Lyric_1"]["@LyricContent"]
    musicinfo = {
        "meta_info": {
            "mediaid": None,
            "media_name": None,
            "singer": None,
        },
        "sub_meata_info": {},
        "clips": [],
    }
    # lyrics = [line.strip() for line in re.split(r"[\t\n\s+]", lyrics)]
    lyrics = ["[" + line.strip() for line in re.split(r"\[", lyrics)]
    next_is_title_row = False
    lyric_clips = []
    for line in lyrics:
        if is_songname(line):
            musicinfo["meta_info"]["media_name"] = get_songname(line)
            continue
        if is_singer(line):
            musicinfo["meta_info"]["singer"] = get_singer(line)
            continue
        if is_album(line):
            musicinfo["meta_info"]["album"] = get_album(line)
            continue
        is_lyric_row = check_is_lyric_row(line)
        if next_is_title_row:
            next_is_title_row = False
            continue
        # remove tille row
        if not next_is_title_row and re.search(r"\[offset[:：]", line):
            next_is_title_row = True
        if is_lyric_row and re.match(r"\[\d+,\d+\]", line):
            lyric_clip = lyric2clip(line)
            lyric_clips.append(lyric_clip)
            clips = lyric2clips(line)
            musicinfo["clips"].extend(clips)
    musicinfo["meta_info"]["lyric"] = lyric_clips
    return musicinfo


def lrc_timestr2time(time_str: str) -> float:
    """提取lrc中的时间戳文本，类似[00:00.00]，转化成秒的浮点数

    Args:
        time_str (str):

    Returns:
        float: 时间浮点数
    """
    m, s, ms = (float(x) for x in re.split(r"[:.]", time_str))
    return round((m * 60 + s + ms / 1000), 3)


def get_lrc_line_time(text: str, time_pattern: str) -> str:
    """提取lrc中的时间字符串, 类似 \"[00:00.00]本字幕由天琴实验室独家AI字幕技术生成\"

    Args:
        text (str): 输入文本
        time_pattern (str): 时间字符串正则表达式

    Returns:
        str: 符合正则表达式的时间信息文本
    """
    time_str = re.search(time_pattern, text).group(0)
    return lrc_timestr2time(time_str)


def lrc_lyric2clip(lyric: str, time_pattern: str, duration: float) -> dict:
    """将一行lrc文本字符串转化为Clip 字典

    Args:
        lyric (str):  类似 \"[00:00.00]本字幕由天琴实验室独家AI字幕技术生成\"
        time_pattern (str): 时间字符串正则表达式，类似 r"\d+:\d+\.\d+"
        duration (float): clip的时长信息，

    Returns:
        dict: 转化后Clip
            Clip定义可以参考 https://git.woa.com/innovative_tech/VideoMashup/blob/master/videomashup/media/clip.py
    """
    time_str = get_lrc_line_time(lyric, time_pattern=time_pattern)
    text = re.sub(time_pattern, "", lyric)
    text = text[2:]
    clip = {
        "time_start": time_str,
        "duration": duration,
        "text": text,
        "timepoint_type": -1,
    }
    return clip


def lrc2musicinfo(lyric: str, time_pattern: str = "\d+:\d+\.\d+") -> dict:
    """将lrc转化为音乐谱面

    Args:
        lyric (str): lrc文本路径
        time_pattern (str, optional): lrc时间戳字符串正则表达式. Defaults to "\d+:\d+\.\d+".

    Returns:
        dict: 生成的音乐谱面字典，定义可参考 https://git.woa.com/innovative_tech/VideoMashup/blob/master/videomashup/music/music_info.py
    """
    if isinstance(lyric, str):
        if os.path.isfile(lyric):
            with open(lyric, "r") as f:
                lyric = [line.strip() for line in f.readlines()]
            return lrc2musicinfo(lyric)
        else:
            lyric = lyric.split("\n")
            return lrc2musicinfo(lyric)
    else:
        musicinfo = {
            "meta_info": {
                "mediaid": None,
                "media_name": None,
                "singer": None,
            },
            "sub_meata_info": {},
            "clips": [],
        }
        # lyrics = [line.strip() for line in re.split(r"[\t\n\s+]", lyrics)]
        lyric_clips = []
        rows = len(lyric)
        for i, line in enumerate(lyric):
            if is_songname(line):
                musicinfo["meta_info"]["media_name"] = line[4:-1]
                continue
            if is_singer(line):
                musicinfo["meta_info"]["singer"] = line[4:-1]
                continue
            if is_album(line):
                musicinfo["meta_info"]["album"] = line[4:-1]
                continue
            if len(re.findall(time_pattern, line)) > 0:
                if i < rows - 1:
                    time_start = get_lrc_line_time(line, time_pattern=time_pattern)
                    next_line_time_start = get_lrc_line_time(
                        lyric[i + 1], time_pattern=time_pattern
                    )
                    duration = next_line_time_start - time_start
                else:
                    duration = 1
                clip = lrc_lyric2clip(
                    line, duration=duration, time_pattern=time_pattern
                )
                musicinfo["clips"].append(clip)
        musicinfo["meta_info"]["lyric"] = lyric_clips
        return musicinfo


def lyricfile2musicinfo(path: str) -> dict:
    """将歌词文件转化为音乐谱面，歌词文件可以是QRC的xml文件、也可以是lrc对应的lrc文件
        TODO： 待支持osu

    Args:
        path (str): 歌词文件路径

    Returns:
        dict: 音乐谱面字典，定义可参考 https://git.woa.com/innovative_tech/VideoMashup/blob/master/videomashup/music/music_info.py
    """

    filename, ext = os.path.basename(path).split(".")
    if ext == "xml":
        lyric = read_xml2json(path)
        musicinfo = lyric2musicinfo(lyric)
    elif ext == "lrc":
        musicinfo = lrc2musicinfo(path)
    musicinfo["meta_info"]["mediaid"] = filename
    return musicinfo
