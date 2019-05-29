import pickle
from utility import create_directory

def pickle_import(filename_in):
    pickle_off = open(filename_in,"rb")
    data = pickle.load(pickle_off)
    pickle_off.close()
    return data

def pickle_export(dirname_out, filename_out, data):
    create_directory(dirname_out)
    pickle_on = open(filename_out,"wb")
    pickle.dump(data, pickle_on)
    pickle_on.close()