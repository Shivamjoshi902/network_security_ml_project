from network_security.exception.exception import custom_exception
from network_security.logging.logger import logging
import yaml
import sys
import os

def read_yaml_file(file_path)->dict:
    try:
        with open(file_path, 'rb') as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise custom_exception(e, sys)
    
def write_yaml_file(file_path: str, content:object, replace: bool = False):
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            yaml.dump(content, file)
    except Exception as e:
        raise custom_exception(e, sys)
