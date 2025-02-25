import blosc2
import os

def bl2_save(array, path):
    # current ver. assumes parent directory exists
    blosc2.save_tensor(array, path, mode='w')

def bl2_load(path):
    if not os.path.exists(path):
        print('File does not exist')
        return None
    else:
        return blosc2.load_tensor(path, mode='r')