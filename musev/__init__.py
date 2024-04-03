import os
import logging
import logging.config

logging.config.fileConfig(os.path.join(os.path.dirname(__file__), "logging.conf"))

logger = logging.getLogger("musev")
