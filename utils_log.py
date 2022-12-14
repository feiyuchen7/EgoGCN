import os
import subprocess

'''
How to use

import config as argumentparser
from datetime import datetime
from utils_log import Logger

TIMESTAMP = "{0:%m-%d/T%H-%M-%S}".format(datetime.now())
log_name = log_name + TIMESTAMP + '.txt'
logger = Logger(log_name)

config = argumentparser.ArgumentParser()
logger.write(config.__repr__())

s_time = datetime.now()
logger.write(any_string_you_want)
e_time = datetime.now()
logger.write('Total time: ' + str(e_time-s_time))

show_gpu('GPU memory usage:', config.gpu)
'''

class Logger(object):
    def __init__(self, output_name):
        dir_name = os.path.dirname(output_name)
        if not os.path.exists(dir_name):
            try:
                os.mkdir(dir_name)
            except Exception as e:
                os.makedirs(dir_name)

        self.log_file = open(output_name, 'w')
        self.infos = {}

    def append(self, key, val):
        vals = self.infos.setdefault(key, [])
        vals.append(val)

    def log(self, extra_msg=''):
        msgs = [extra_msg]
        for key, vals in self.infos.iteritems():
            msgs.append('%s %.6f' % (key, np.mean(vals)))
        msg = '\n'.join(msgs)
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        self.infos = {}
        return msg

    def write(self, msg):
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        print(msg)

def show_gpu(msg, gpu_num):
    """
    ref: https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4
    """
    def query(field):
        return(subprocess.check_output(
            ['nvidia-smi', f'--query-gpu={field}',
                '--format=csv,nounits,noheader'], 
            encoding='utf-8'))
    def to_int(result, gpu_num):
        return int(result.strip().split('\n')[gpu_num])
    
    used = to_int(query('memory.used'), gpu_num)
    total = to_int(query('memory.total'), gpu_num)
    pct = used/total
    print('\n' + f'[gpu:{gpu_num}] ' + msg, f'{100*pct:2.1f}% ({used} MiB out of {total}) MiB')  