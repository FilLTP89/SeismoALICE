import logging
import logging.config
from pathlib import Path




def setup_logging( log_config = 'logger/logger_config.log',
        logging_level   = logging.INFO, 
        default_level   = logging.DEBUG, 
        logging_format  = '%(levelname)s:%(message)s'):
    
    logger  = logging.getLogger(__name__)
    logger.setLevel(logging_level)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    formatter   = logging.Formatter(logging_format)
    log_config  = Path(log_config)

    if log_config.is_file():
        file_handler = logging.FileHandler(log_config)

        file_handler.setLevel(logging_level)
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)

    return logger