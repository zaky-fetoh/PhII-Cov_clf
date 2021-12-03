
import tarfile as tar


def dataset_extraction(file_name,
                       outpath):
    with tar.open(file_name) as file:
        file.extractall(outpath)