import os


def kmp_duplicate_lib_ok(on: bool = True):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" if on else "FALSE"


def basename_noext(p):
    return os.path.splitext(os.path.basename(p))[0]
