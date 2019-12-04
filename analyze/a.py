import os, glob
import json
import pdb

class CreateData():
    def __init__(self, datafolder, output_folder):
        self._datafolder = datafolder
        self._output_folder = output_folder
        self._classes = []

        self._process()

    def _process(self):
        for each_mode in ['train', 'val']:
            self._process_mode(each_mode)

    def _process_mode(self, mode):



if __name__=="__main__":

    ############################################################
    datafolder = '/home/ubuntu/cr_data/resize_kinetics400'
    #datafolder = '/home/ubuntu/cr_data/kinetics400'
    output_folder = '/home/ubuntu/slowfast/SlowFast/data_processing'
    ############################################################


    create_data = CreateData(datafolder, output_folder)
