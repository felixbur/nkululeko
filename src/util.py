# util.py
import audeer
import ast 
import sys
import glob_conf

class Util:
        
    def get_path(self, entry):
        root = glob_conf.config['EXP']['root']
        name = glob_conf.config['EXP']['name']
        try:
            entryn = glob_conf.config['EXP'][entry]
        except KeyError:
            if entry == 'fig_dir':
                entryn = './images/'
            elif entry == 'res_dir':
                entryn = './results/'
            else:
                entryn = './store/'
        dir_name = f'{root}{name}/{entryn}'
        audeer.mkdir(dir_name)
        return dir_name

    def get_exp_name(self):
        ds = '_'.join(ast.literal_eval(glob_conf.config['DATA']['databases']))
        mt = glob_conf.config['MODEL']['type']
        ft = glob_conf.config['FEATS']['type']
        return f'{ds}_{mt}_{ft}'

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
            return glob_conf.config[category][key]
        except KeyError:
            return default