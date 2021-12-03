import zipfile

def dataset_extraction(file_name, outpath):
    with zipfile.ZipFile(file_name) as file:
        file.extractall(outpath)