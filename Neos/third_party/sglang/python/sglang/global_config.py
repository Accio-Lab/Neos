"""Global configurations"""




class GlobalConfig:
    """
    Store some global constants.
    """

    def __init__(self):
        
        
        
        self.verbosity = 0

        
        self.default_backend = None

        
        self.skip_special_tokens_in_output = True
        self.spaces_between_special_tokens_in_out = True

        
        self.enable_precache_with_tracing = True
        self.enable_parallel_encoding = True


global_config = GlobalConfig()
