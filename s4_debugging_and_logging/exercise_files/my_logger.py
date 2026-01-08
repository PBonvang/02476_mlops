import sys
from loguru import logger
from logging import WARNING

log_level = WARNING
logger.remove()  # Remove the default logger
logger.add(sys.stdout, level=log_level)  # Add a new logger with WARNING level
logger.add("my_log.log", level=log_level, rotation="100 MB")  # Log all levels to a file

logger.debug("Used for debugging your code.")
logger.info("Informative messages from your code.")
logger.warning("Everything works but there is something to be aware of.")
logger.error("There's been a mistake with the process.")
logger.critical("There is something terribly wrong and process may terminate.")