import pathlib
import datetime
import json
from numpy import shape
from numpy import savez

class Logger:
    def __init__(self,
        description):

    	self.path = '../logs/{}/'.format(description)
    	pathlib.Path(self.path).mkdir(parents=True, exist_ok=True)
    	self.results_file = open('{}log_autoenc_loss.csv'.format(self.path), 'w+')
    	#self.config_file.write(json.dumps(params))
    	#self.config_file.close()

    @staticmethod
    def _dump_to_file(filename,
        vals): 
        savez(filename, **vals)

    def dump_res(self,
        vals,
        vals_description=['original', 
        'encoded', 
        'squeezed', 
        'decoded', 
        #'labels', 
        'second_set',
        'second_set_original'],
        attr='train'):

        for ind, el in enumerate(vals):
            # add labels to data for convinience
            print(len(el))
            data = {key : el for key, el in zip(vals_description, el)}
            self._dump_to_file('{}model{}_res_{}'.format(self.path, ind, attr), data)

    def log_results(self,
        epoch,
        losses):

    	self.results_file.write('%i %s\n' % (epoch, losses))

    def __exit__(self,
        type,
        value,
        traceback):
    
        self.results_file.close()

    def __enter__(self):
        return self
