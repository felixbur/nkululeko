# util.py
import audeer
import ast 
import sys

class Util:
    def __init__(self, config):
        self.config = config

    def get_path(self, entry):
        root = self.config['EXP']['root']
        name = self.config['EXP']['name']
        entryn = self.config['EXP'][entry]
        dir_name = f'{root}{name}/{entryn}'
        audeer.mkdir(dir_name)
        return dir_name

    def exp_is_classification(self):
        if self.config['EXP']['type']=='classification':
            return True
        return False

    def error(self, message):
        print(f'ERROR: {message}')
        sys.exit()