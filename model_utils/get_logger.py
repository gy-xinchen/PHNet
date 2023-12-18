import logging
import os

def get_logger(file_path):
    log_directory = os.path.dirname(file_path)
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    logger = logging.getLogger("Custom logger name")
    if not logger.handlers:
        logger.setLevel("INFO")
        console_handler = logging.StreamHandler()
        file_hadler = logging.FileHandler(file_path, mode="a", encoding="UTF-8")
        console_handler.setFormatter(logging.Formatter("%(asctime)s   %(message)s"))
        file_hadler.setFormatter(logging.Formatter("%(asctime)s   %(message)s"))
        logger.addHandler(console_handler)  # Pass in the console handler
        logger.addHandler(file_hadler)

    return logger