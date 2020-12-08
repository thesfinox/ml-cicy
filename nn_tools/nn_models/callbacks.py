from tensorflow import keras
import numpy as np
import time

class PrintCheckpoint(keras.callbacks.Callback):
    '''
    Print status of the training after a given number of epochs.
    '''
    
    def __init__(self, interval):
        '''
        Arguments:
            interval: print status after given interval of epochs.
        '''
        super(PrintCheckpoint, self).__init__()
        self.interval    = int(interval)
        self.epoch_times = []
        
    def on_train_begin(self, logs=None):
        if self.interval > 1:
            print(f'Training has started. Callouts will be printed every {self.interval:d} epochs.', flush=True)
        else:
            print(f'Training has started. Callouts will be printed every epoch.', flush=True)
                
        self.train_time = time.time()
        
    def on_train_end(self, logs=None):
        self.train_time = time.time() - self.train_time
        self.train_time = time.gmtime(self.train_time)
        self.train_time = time.strftime('%H hours, %M minutes, %S seconds', self.train_time)
        now = time.strftime('%d/%m/%Y at %H:%M:%S', time.localtime())
        print(f'\nTraining ended on {now} after {self.train_time}.', flush=True)
        
    def on_epoch_begin(self, epoch, logs=None):
        
        if (epoch + 1) % self.interval == 0 or epoch == 0:
            self.epoch_time = time.time()
            now = time.strftime('%d/%m/%Y at %H:%M:%S', time.localtime())
            print(f'\nTraining epoch {epoch+1:d}. Started on {now}.\n', flush=True)
        
    def on_epoch_end(self, epoch, logs=None):
        
        if (epoch + 1) % self.interval == 0 or epoch == 0:
            self.epoch_time = time.time() - self.epoch_time
            self.epoch_times.append(self.epoch_time)
            epoch_time = time.strftime('%H hours, %M minutes, %S seconds', time.gmtime(np.mean(self.epoch_times)))
            print(f'    Average epoch training time: {epoch_time}\n', flush=True)
            for key, value in logs.items():
                print(f'    {key} = {value:.6f}')