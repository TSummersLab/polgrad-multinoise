import os
import sys

def inplace_print(printstr):
    delete = '\b' * len(printstr)
    print("{0}{1}".format(delete, printstr), end="")
    sys.stdout.flush()

def printout(string,file):
    print(string)
    file.write(string+'\n')

def create_directory(dirname_out):
    # Create target directory & all intermediate directories if nonexistent
    if not os.path.exists(dirname_out):
        os.makedirs(dirname_out)
        print("Directory '%s' created" % dirname_out)