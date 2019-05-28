import pickle
from time import time,sleep
from utility import create_directory

def pickle_import(filename_in):
    # Import
#    print('Starting data import...')
#    start_time = time()
    pickle_off = open(filename_in,"rb")
    data = pickle.load(pickle_off)
    pickle_off.close()
#    end_time = time()
#    print('Data import completed in %.2f seconds' % (end_time-start_time))
#    sleep(0.01)
    return data


def pickle_export(dirname_out, filename_out, data):
    # Export
#    print('Starting data export...')
#    start_time = time()
    create_directory(dirname_out)
    pickle_on = open(filename_out,"wb")
    pickle.dump(data, pickle_on)
    pickle_on.close()
#    end_time = time()
#    print('Data export completed in %.2f seconds' % (end_time-start_time))
#    sleep(0.01)