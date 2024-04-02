

class SubprocessError(Exception):
    """
    Exception object that contains information about an error that occurred
    when running a command line command with a subprocess.
    """

    def __init__(self, cmd, return_code, stdout, stderr, *args):
        msg = 'Got non-zero exit code ({1}) from command "{0}": {2}'
        if stderr.strip():
            err_msg = stderr
        else:
            err_msg = stdout
        msg = msg.format(cmd[0], return_code, err_msg)
        self.cmd = cmd
        self.cmd_return_code = return_code
        self.cmd_stdout = stdout
        self.cmd_stderr = stderr
        super(SubprocessError, self).__init__(msg, *args)
