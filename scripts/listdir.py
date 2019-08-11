from os import listdir
from os.path import isfile, join

def list_files(path, sub_str=''):
    return [(join(path, f), f) for f in listdir(path) if isfile(join(path, f)) and sub_str in f]