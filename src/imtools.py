import os

def get_imlist(path):
    """return a list of filenames for all images in a dir"""
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
