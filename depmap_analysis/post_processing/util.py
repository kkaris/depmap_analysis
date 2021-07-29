from pathlib import Path
from typing import List, Optional
from indra_db.util.s3_path import S3Path
from indra.util.aws import get_s3_client


def get_dir_iter(path: str, file_ending: Optional[str] = None) -> List:
    """Takes a directory path and returns a list of files

    Parameters
    ----------
    path :
        The path to the directory to loop over
    file_ending :
        If provided, files in the returned list must be of this format,
        e.g. .pkl

    Returns
    -------
    :
        A list of file in the directory
    """
    if path.startswith('s3://'):
        s3 = get_s3_client(unsigned=False)
        s3_base_path = S3Path.from_string(path)
        input_iter = \
            [s3p.to_string() for s3p in s3_base_path.list_objects(s3)]
    else:
        local_base_path = Path(path)
        input_iter = [f.absolute().as_posix() for f in
                      local_base_path.glob('*') if f.is_file()]

    if file_ending:
        input_iter = [f for f in input_iter if f.endswith(file_ending)]

    return input_iter
