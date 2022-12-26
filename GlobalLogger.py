import sys
import logging


logger = logging.getLogger("toto")

def initializeLogger(level = logging.WARNING):
    console_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(console_handler)
    logger.setLevel(level)