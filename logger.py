import pathlib
import datetime
import json
from numpy import savetxt

class Logger:
    def __init__(self, description):
    	self.path = '../logs/{}/'.format(description)
    	pathlib.Path(self.path).mkdir(parents=True, exist_ok=True)
    	self.results_file = open('{}log_autoenc_loss.csv'.format(self.path), 'w+')
    	#self.config_file.write(json.dumps(params))
    	#self.config_file.close()

    @staticmethod
    def _dump_to_file(filename, var):
    	savetxt('{}{}.txt'.format(filename, datetime.datetime.now()), var)

    def dump_res(self, valid_pred, valid_orig, train_pred, train_orig):
    	self._dump_to_file('{}dump_res_autoenc_pr_valid'.format(self.path), valid_pred)
    	#self._dump_to_file('{}dump_res_autoenc_or_valid'.format(self.path), valid_orig)
    	self._dump_to_file('{}dump_res_autoenc_pr_train'.format(self.path), train_pred)
    	#self._dump_to_file('{}dump_res_autoenc_or_train'.format(self.path), train_orig)

    def log_results(self, epoch, losses):
    	self.results_file.write('%i %s\n' % (epoch, losses))

    def __exit__(self, type, value, traceback):
        #self.config_file.close()
        self.results_file.close()

    def __enter__(self):
        return self
