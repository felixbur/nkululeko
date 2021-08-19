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
        type = self.config_val('EXP', 'type', 'classification')
        if type=='classification':
            return True
        return False

    def error(self, message):
        print(f'ERROR: {message}')
        sys.exit()

    def debug(self, message):
        print(f'DEBUG: {message}')

    def config_val(self, category, key, default):
        try:
            # strategy is either train_test (default)  or cross_data
            return self.config[category][key]
        except KeyError:
            return default