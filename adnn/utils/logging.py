import logging

def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Configure logger only once
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger 