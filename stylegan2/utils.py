import sys
import time
import random
from pathlib import Path

class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax.
    From https://github.com/NVlabs/stylegan3/blob/main/dnnlib/util.py"""

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value):
        self[name] = value

    def __delattr__(self, name: str):
        del self[name]
        
        
def create_experiment_folder(path_results, folder_name):
    ''' 
    Creates a folder to save experiment results and parameters 
    Args:
        path_results: where to save
    '''
    
    if not path_results.exists():
        print(f'Path "{path_results}" does not exist, need to make directory.')
        path_results.mkdir()
    
    
    # wait a random time to avoid problems with parallel calls
    time.sleep(1*random.random())
    
    timestamp = time.strftime('%Y-%m-%d_%H%M', time.localtime())
    path_results_exp = Path(path_results) / (timestamp + f'_{folder_name}')
    path_results_exp_i = path_results_exp
    i = 1
    while path_results_exp_i.exists():
        parts_i = list(path_results_exp.parts)
        parts_i[-1] += f'_({i})'
        path_results_exp_i = Path(*parts_i)
        i += 1
        
    path_results_exp_i.mkdir()
    
    return path_results_exp_i


class Logger(object):
    """Redirect stderr to stdout, optionally print stdout to a file, 
    and optionally force flushing on both stdout and the file.
    From https://github.com/NVlabs/stylegan3/blob/main/dnnlib/util.py"""

    def __init__(self, file_name: str = None, file_mode: str = "w", should_flush: bool = True):
        self.file = None

        if file_name is not None:
            self.file = open(file_name, file_mode)

        self.should_flush = should_flush
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        sys.stdout = self
        sys.stderr = self

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def write(self, text) -> None:
        """Write text to stdout (and a file) and optionally flush."""
        if isinstance(text, bytes):
            text = text.decode()
        if len(text) == 0: # workaround for a bug in VSCode debugger: sys.stdout.write(''); sys.stdout.flush() => crash
            return

        if self.file is not None:
            self.file.write(text)

        self.stdout.write(text)

        if self.should_flush:
            self.flush()

    def flush(self) -> None:
        """Flush written text to both stdout and a file, if open."""
        if self.file is not None:
            self.file.flush()

        self.stdout.flush()

    def close(self) -> None:
        """Flush, close possible files, and remove stdout/stderr mirroring."""
        self.flush()

        # if using multiple loggers, prevent closing in wrong order
        if sys.stdout is self:
            sys.stdout = self.stdout
        if sys.stderr is self:
            sys.stderr = self.stderr

        if self.file is not None:
            self.file.close()
            self.file = None