import logging
import logging.config
from pathlib import Path
from utils import read_json



def setup_logging(save_dir, 
            log_config      = 'logger/logger_config.json', 
            default_level   =  logging.INFO, 
            logging_format  =  '%(levelname)s:%(message)s'):
    
    logger  = logging.getLogger(__name__)
    logger.setLevel(default_level)

    formatter   = logging.Formatter(LOGGING_FORMAT)
    log_config  = Path(log_config)

    if log_config.is_file():
        file_handler = logging.FileHandler(log_config)

        file_handler.setLevel(LOGGING_LEVEL)
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)

    return logger