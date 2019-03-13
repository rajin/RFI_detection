

''' The main purpose of this code is to apply various processing procedures
to the data Radio data that will be fed to the RFI detection algorithm.
This includes simulated data or real observations 
'''


import numpy as np 
import h5py


def open_h5file(filename):
	return h5py.File(filename,'r')

def get_data_arr(filename):
	data = open_h5file(filename)
	group = data['output']
	return group['vis_clean'].value,group['vis_dirty'].value
