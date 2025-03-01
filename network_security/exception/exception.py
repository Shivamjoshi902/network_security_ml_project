import logging
import sys
from network_security.logging.logger import logging

def error_msg_detail(error, error_detail:sys):
    _,_,err_tb = error_detail.exc_info()
    file_name = err_tb.tb_frame.f_code.co_filename
    error_msg = "error occured in python scrypt , error name : [{0}] , line number : [{1}] error message [{2}]".format(file_name, err_tb.tb_lineno, str(error)) 
    return error_msg


class custom_exception(Exception):
    def __init__(self, error_msg, error_detail:sys):
        super().__init__(error_msg)
        self.error_msg = error_msg_detail(error_msg, error_detail)
    
    def __str__(self):
        return self.error_msg    