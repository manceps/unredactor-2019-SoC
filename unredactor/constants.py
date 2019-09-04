import os
import logging


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
if not os.path.isdir(DATA_DIR):
    os.mkdir(DATA_DIR)
LOG_DIR = os.path.join(DATA_DIR, 'log')
if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)


LOG_LEVELS = [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.FATAL]
LOG_LEVEL_NAMES = 'DEBUG INFO WARNING ERROR FATAL'.split()
LOG_LEVEL_ABBREVIATIONS = [s[:4].lower() for s in LOG_LEVEL_NAMES]

root_logger = logging.getLogger()
root_logger.setLevel(logging.WARN)

log = logging.getLogger(__name__)
handler = logging.handlers.TimedRotatingFileHandler(LOG_DIR, when='midnight')
handler.setLevel(logging.INFO)
# formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# handler.setFormatter(formatter)
# handler.suffix = "%Y-%m-%d.log"
log.addHandler(handler)
log.setLevel(logging.INFO)
