# encoding: utf-8
"""UI functions"""

import sys


def progressbar(it, prefix="", size=60, file=sys.stdout):
    """ Iterable progress bar.

    """
    count = len(it)

    def show(j):
        x = int(size*j/count)
        file.write("'\r%s[%s%s] %i/%i\r" %
                   (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()
