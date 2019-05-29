import sys

def inplace_print(printstr):
    delete = '\b' * len(printstr)
    print("{0}{1}".format(delete, printstr), end="")
    sys.stdout.flush()