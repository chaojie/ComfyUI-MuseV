import os

from .ffmpeg import ffmpeg_load


def extract_flickr_id(url):
    return url.strip('/').split('/')[-4]


def download_flickr(url: str, save_path: str) -> str:
    code = -1
    code = ffmpeg_load(url=url,
                       save_path=save_path)
    if code == 0:
        return (code, save_path)
    # only retry when failed!
    flickr_id = extract_flickr_id(url)
    url = 'https://www.flickr.com/video_download.gne?id={}'.format(
        flickr_id)
    code = ffmpeg_load(url=url,
                       save_path=save_path)
    return save_path