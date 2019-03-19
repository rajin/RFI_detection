
''' The main purpose of this code is to apply various processing procedures
to the data Radio data that will be fed to the RFI detection algorithm.
This includes simulated data or real observations 
'''


import numpy as np 
import h5py
import pylab as plt 


def open_h5file(filename):
	#function to read HDF5 file format
	return h5py.File(filename,'r')

def get_data_arr(filename):
	#function that returns clean and dirty array from simulated data
	data = open_h5file(filename)
	group = data['output']
	return group['vis_clean'].value,group['vis_dirty'].value

def save_h5file(savename,input_array,dataset='dataset'):
	#function to save hdf5 
	out = h5py.File(savename, 'w')
	out.create_dataset(dataset, data=input_array)
	return

def analogue_2_binary(input_array,threshold):
	#function to turn analogue array to a binary array consisting of only 0's and 1's
	return (input_array > threshold).astype(int)

def process_complex_arr(arr_in,output_format='absolute magnitude'):
	#function to select different parts of a complex array
	if output_format == 'absolute magnitude':
		return np.abs(arr_in)
	if output_format == 'real part':
		return np.real(arr_in)
	if output_format == 'imaginary part':
		return np.imag(arr_in)

def scale_array(arr_in,lower_lim=None,upper_lim=None):
	#scale array between 0 and 1, By default maximum value is used to scale the array
	# if upper_lim given is less than np.max(arr_in) then it will effectively clip
	# the maximum value
	if upper_lim == None:
		upper_lim = np.max(arr_in)
	if lower_lim == None:
		lower_lim = np.min(arr_in)
	arr_in[arr_in>=upper_lim] = upper_lim
	arr_in[arr_in<=lower_lim] = lower_lim
	arr_in = arr_in-lower_lim
	arr_in = arr_in/upper_lim
	return arr_in


def data_proc_1(filename):
	#Data processing sequence 
	clean,dirty = get_data_arr(filename)
	#clean_1 = process_complex_arr(clean)
	dirty_1 = process_complex_arr(dirty)
	dirty_2 = scale_array(dirty_1)
	#rfi_1 = dirty_1-clean_1
	rfi = dirty-clean 
	rfi_1 = process_complex_arr(rfi)
	rfi_2 = scale_array(rfi_1)
	return dirty_2,rfi_2


def create_im_sequences(arr_in,seq_length=8):
	#function to reshape array to have a list of sequences 
	#arr_in must be of the format (a,b,c,d) where
	#a is layers of simulated data with different settings
	#b is the time steps
	#c and d are the image rows and columns
	num_of_seq = arr_in.shape[1]//seq_length
	total_num_seq = num_of_seq*arr_in.shape[0]
	arr_out = np.zeros((total_num_seq,seq_length,arr_in.shape[2],arr_in.shape[3]),float)
	count = 0
	for i in range(arr_in.shape[0]):
		for j in range(num_of_seq):
			arr_out[count] = arr_in[i,j*seq_length:(j+1)*seq_length,:,:]
			count += 1
	return arr_out


def pipeline1(filenames):
	#function to process data in a given specific sequence. 
	#input should be a list of HDF5 filenames of simulated data
	dummy_clean,dummy_dirty = get_data_arr(filenames[0])
	dirty_arr = np.zeros((len(filenames),dummy_dirty.shape[0],dummy_dirty.shape[1],dummy_dirty.shape[2],dummy_dirty.shape[3]),float)
	rfi_arr = np.zeros((len(filenames),dummy_dirty.shape[0],dummy_dirty.shape[1],dummy_dirty.shape[2],dummy_dirty.shape[3]),float)
	for i in range(len(filenames)):
		dummy_dirty,dummy_rfi = data_proc_1(filenames[i])
		dirty_arr[i] = dummy_dirty
		rfi_arr[i] = dummy_rfi

	dirty_seq = create_im_sequences(dirty_arr[:,:,:,:,0])
	rfi_seq = create_im_sequences(rfi_arr[:,:,:,:,0])

	save_h5file('dirty_seq.h5',dirty_seq,dataset='dirty_seq')
	save_h5file('rfi_seq.h5',rfi_seq,dataset='rfi_seq')
	return dirty_seq,rfi_seq




