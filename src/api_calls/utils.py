from typing import Iterable

def batch(iterable: Iterable, n=1) -> Iterable:
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]
