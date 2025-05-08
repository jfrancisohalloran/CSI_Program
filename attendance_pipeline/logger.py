import logging
import os

def configure_logging(log_file: str = None):
    fmt = "%(asctime)s %(levelname)s [%(name)s]: %(message)s"
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        handlers.append(fh)
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)

def get_logger(name: str):
    return logging.getLogger(name)
