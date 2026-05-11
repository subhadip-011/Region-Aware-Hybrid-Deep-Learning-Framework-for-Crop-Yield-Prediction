# src/utils/exception.py

import sys
from src.utils.logger import logger


# =====================================================
# ERROR MESSAGE FUNCTION
# =====================================================

def error_message_detail(error, error_detail):
    """
    Generate detailed error message
    """

    _, _, exc_tb = error_detail.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename

    line_number = exc_tb.tb_lineno

    error_message = (
        f"\nError occurred in script: [{file_name}]"
        f"\nLine number: [{line_number}]"
        f"\nError message: [{str(error)}]"
    )

    return error_message


# =====================================================
# CUSTOM EXCEPTION CLASS
# =====================================================

class CustomException(Exception):
    """
    Custom exception class for project
    """

    def __init__(self, error_message, error_detail):

        super().__init__(error_message)

        self.error_message = error_message_detail(
            error_message,
            error_detail
        )

        logger.error(self.error_message)

    def __str__(self):

        return self.error_message