import sys

#def inplace_print(a,digits=18,decimals=9,suffix_str):
#    delete = "\b" * (digits)
#    print("{0}{1:{2}.{3}f}".format(delete, a, digits, decimals), end="")
#    sys.stdout.flush()
#    
    
def inplace_print(printstr):
    delete = '\b' * len(printstr)
    print("{0}{1}".format(delete, printstr), end="")
    sys.stdout.flush()