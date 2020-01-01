import os
import logging

LOG_BASE_NAME = 'pygst'
LOG_FORMAT = '%(levelname)-6.6s | %(name)-20s | %(asctime)s.%(msecs)03d | %(threadName)s | %(message)s'
LOG_DATE_FORMAT = '%d.%m %H:%M:%S'
GST_PYTHON_LOG_LEVEL = int(os.getenv("GST_PYTHON_LOG_LEVEL", logging.DEBUG / 10)) * 10


def setup_logging(verbose: int = logging.DEBUG, name=None):
    """Configure console logging. Info and below go to stdout, others go to stderr. """

    root_logger = logging.getLogger(name)
    root_logger.setLevel(logging.DEBUG if verbose > 0 else logging.INFO)

    log_handler = logging.StreamHandler()
    log_handler.setFormatter(logging.Formatter(
        fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT))

    local_logger = logging.getLogger(LOG_BASE_NAME)
    local_logger.setLevel(verbose)

    root_logger.addHandler(log_handler)


setup_logging(GST_PYTHON_LOG_LEVEL)
