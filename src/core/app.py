class SingletonMeta(type):
    """
    Implementation of the singleton pattern 
    """

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
            cls._setup =  session.setup
        return cls._instances[cls]
