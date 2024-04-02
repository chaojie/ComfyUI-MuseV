
import os

from pytube import YouTube


def download_youtube(url, format, save_dir, filename):
    youtube = YouTube(url)
    streams = youtube.streams.filter(progressive=True,
                                     file_extension=format)
    save_path = streams.get_highest_resolution().download(output_path=save_dir,
                                              filename=filename)
    return save_path