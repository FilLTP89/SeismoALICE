# -*- coding: utf-8 -*-
#!/usr/bin/env python3
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
        def profile(func): return func
