import subprocess

from .error import SubprocessError


class FfmpegInvalidURLError(Exception):
    """
    Exception raised when a 4XX or 5XX error is returned when making a request
    """

    def __init__(self, url, error, *args):
        self.url = url
        self.error = error
        msg = 'Got error when making request to "{}": {}'.format(url, error)
        super(FfmpegInvalidURLError, self).__init__(msg, *args)


def ffmpeg_load(url: str, save_path: str) -> str:

    def run(cmd):
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate()
        return_code = proc.returncode

        if return_code != 0:
            raise SubprocessError(
                cmd, return_code, stdout.decode(), stderr.decode())
        return return_code

    command = ['ffmpeg', '-n', '-i', url, '-t', '10', '-f', 'mp4',
               '-r', '30', '-vcodec', 'h264', save_path, '-loglevel', 'error']
    code = run(command)
    return code





