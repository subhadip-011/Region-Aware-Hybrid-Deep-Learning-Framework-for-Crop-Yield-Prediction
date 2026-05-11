# src/utils/logger.py

import logging
import os
from datetime import datetime

# =====================================================
# CREATE LOG DIRECTORY
# =====================================================

LOG_DIR = 'logs'

os.makedirs(LOG_DIR, exist_ok=True)

# =====================================================
# LOG FILE NAME
# =====================================================

log_filename = datetime.now().strftime(
    'crop_yield_%Y%m%d_%H%M%S.log'
)

log_filepath = os.path.join(
    LOG_DIR,
    log_filename
)

# =====================================================
# LOGGER CONFIGURATION
# =====================================================

logging.basicConfig(

    level=logging.INFO,

    format=(
        '%(asctime)s | '
        '%(levelname)s | '
        '%(name)s | '
        '%(message)s'
    ),

    datefmt='%Y-%m-%d %H:%M:%S',

    handlers=[

        logging.FileHandler(
            log_filepath,
            encoding='utf-8'
        ),

        logging.StreamHandler()
    ]
)

# =====================================================
# LOGGER OBJECT
# =====================================================

logger = logging.getLogger('CropYieldPrediction')

# =====================================================
# STARTUP MESSAGE
# =====================================================

logger.info(
    'Logger initialized successfully'
)