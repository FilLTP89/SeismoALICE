# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import pstats
from pstats import SortKey

# def read():
#     with open("output_time.txt","w") as f:
#         p = pstats.Stats("output.dat", stream=f)
#         p.sort_stats("output.dat").print_stats()

#     with open("output_calls.txt", "w") as f:
#         p = pstats.Stats("output.dat", stream=f)
#         p.sort_stats("calls").print_stats()
try:
    # Python 2
    import __builtin__ as builtins
except ImportError:
    # Python 3
    import builtins

    try:
        profile = builtins.profile
    except AttributeError:
        # No line profiler, provide a pass-through version
        def profile(func):
            return func(*args, **kwargs)
        # read()
        



 