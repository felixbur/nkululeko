# util.py
import audeer
import ast 
import sys
import glob_conf
import numpy as np

class Util:
        
    def get_path(self, entry):
        root = glob_conf.config['EXP']['root']
        name = glob_conf.config['EXP']['name']
        try:
            entryn = glob_conf.config['EXP'][entry]
        except KeyError:
            # some default values
            if entry == 'fig_dir':
                entryn = './images/'
            elif entry == 'res_dir':
                entryn = './results/'
            elif entry == 'model_dir':
                entryn = './models/'
            else:
                entryn = './store/'

        # Expand image, model and result directories with run index
        if entry == 'fig_dir' or entry == 'res_dir' or entry == 'model_dir':
            run = self.config_val('EXP', 'run', 0)
            entryn = entryn +  f'run_{run}/'

        dir_name = f'{root}{name}/{entryn}'
        audeer.mkdir(dir_name)
        return dir_name
    
    def get_res_dir(self):
        root = glob_conf.config['EXP']['root']
        name = glob_conf.config['EXP']['name']
        dir_name = f'{root}{name}/results/'
        audeer.mkdir(dir_name)
        return dir_name


    def get_exp_name(self):
        ds = '_'.join(ast.literal_eval(glob_conf.config['DATA']['databases']))
        mt = glob_conf.config['MODEL']['type']
        ft = glob_conf.config['FEATS']['type']
        return f'{ds}_{mt}_{ft}'

    def get_plot_name(self):
        try:
            plot_name = glob_conf.config['PLOT']['name']
        except KeyError:
            plot_name = self.get_exp_name()
        return plot_name

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

    def set_config_val(self, section, key, value):
        try:
            # does the section already exists?
            glob_conf.config[section][key] = str(value)
        except KeyError:
            glob_conf.config.add_section(section)
            glob_conf.config[section][key] = str(value)

    def config_val(self, section, key, default):
        try:
            # strategy is either train_test (default)  or cross_data
            return glob_conf.config[section][key]
        except KeyError:
            return default

    def continuous_to_categorical(self, array):
        bins = ast.literal_eval(glob_conf.config['DATA']['bins'])
        result =  np.digitize(array, bins)-1
        return result

    def print_best_results(self, best_reports):
        res_dir = self.get_res_dir()
        # go one level up above the "run" level
        all = ''
        vals = np.empty(0)
        for report in best_reports:
            all += str(report.result.test) + ', '
            vals = np.append(vals, report.result.test)
        file_name = f'{res_dir}{self.get_exp_name()}_runs.txt'
        with open(file_name, "w") as text_file:
            text_file.write(all)
            text_file.write(f'\nmean: {vals.mean():.3f}, max: {vals.max():.3f}, max_index: {vals.argmax()}')
