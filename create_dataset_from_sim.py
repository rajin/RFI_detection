
import numpy as np
import h5py
import glob
from sklearn.utils import shuffle

def open_h5file(filename):
	#function to read HDF5 file format
	return h5py.File(filename,'r')

def read_h5file(filename,dataset='dataset'):
	#function to read a hdf5 file and output the array as a np.array
	data = open_h5file(filename)
	return data[dataset].value

def save_h5file(savename,input_array,dataset='dataset'):
	#function to save hdf5 
	out = h5py.File(savename, 'w')
	out.create_dataset(dataset, data=input_array)
	out.close()
	return

def process_complex_arr(arr_in,output_format='absolute magnitude'):
	#function to select different parts of a complex array
	if output_format == 'absolute magnitude':
		return np.abs(arr_in)
	if output_format == 'real part':
		return np.real(arr_in)
	if output_format == 'imaginary part':
		return np.imag(arr_in)

def create_noise(arr_in,mu,sigma,seed=0):
	#Function to create real and imaginary noise for an array.
	np.random.seed(seed)
	len_arr_in_shape = len(arr_in.shape)
	total_size = 1 
	for i in range(len_arr_in_shape):
		total_size *= arr_in.shape[i]

	create_random_num_arr = np.random.normal(mu,sigma, total_size*2)
	real_part = create_random_num_arr[:total_size].reshape(arr_in.shape)
	im_part = create_random_num_arr[total_size:].reshape(arr_in.shape)

	noise = real_part + 1j*im_part
	return noise

#### loading the data
astro_sources = read_h5file('astro_sources.h5',dataset='astro_sources')
rfi = read_h5file('rfi.h5',dataset='rfi')

#creating noise and creating the dirty visibility
noise_array = create_noise(astro_sources,0,0.168)
dirty_vis = astro_sources + noise_array + rfi

# saving file as backup
save_h5file('sub_sample_dirty_vis.h5',dirty_vis,dataset='sub_sample_dirty_vis')
del astro_sources
del noise_array

###################################### creating first 2D dataset - merging all polarisation into 1 ##########################
#looping over 4 polarization channels
first_data_dirty_vis = np.zeros((dirty_vis.shape[0]*4,100,4096),dtype=np.complex_)
first_data_rfi_vis = np.zeros((dirty_vis.shape[0]*4,100,4096),dtype=np.complex_)
k = 0
for i in range(dirty_vis.shape[0]):
	for j in range(4):
		first_data_dirty_vis[k,:,:] = dirty_vis[i,:,:,j]
		first_data_rfi_vis[k,:,:] = rfi[i,:,:,j]
		k += 1

##### converting complex values to absolute_mag
first_data_dirty_vis_mag = process_complex_arr(first_data_dirty_vis,output_format='absolute magnitude')
first_data_rfi_vis_mag = process_complex_arr(first_data_rfi_vis,output_format='absolute magnitude')

#### shuffling dataset to be able to train test etc 
#creating shuffle indices
sim_im_num = first_data_dirty_vis_mag.shape[0]
indices = np.arange(0,sim_im_num)
shuffle_indices = shuffle(indices, random_state=0)

#applying shuffling
first_data_dirty_vis_mag_shuffle = np.zeros(first_data_dirty_vis_mag.shape,float)
first_data_rfi_vis_mag_shuffle = np.zeros(first_data_rfi_vis_mag.shape,float)
for i in range(first_data_dirty_vis_mag.shape[0]):
	first_data_dirty_vis_mag_shuffle[i,:,:] = first_data_dirty_vis_mag[shuffle_indices[i],:,:]
	first_data_rfi_vis_mag_shuffle[i,:,:] = first_data_rfi_vis_mag[shuffle_indices[i],:,:]

########## saving dataset for first case 
save_h5file('shuf_firstdat_dirty_vis.h5',first_data_dirty_vis_mag_shuffle,dataset='shuf_firstdat_dirty_vis')
save_h5file('shuf_firstdat_rfi_vis.h5',first_data_rfi_vis_mag_shuffle,dataset='shuf_firstdat_rfi_vis')

#clearing memory
del first_data_dirty_vis_mag_shuffle
del first_data_rfi_vis_mag_shuffle
del first_data_dirty_vis_mag
del first_data_rfi_vis_mag
del first_data_dirty_vis
del first_data_rfi_vis

###################################### creating 2nd dataset calculating stokes parameters ##########################

def create_stokes_from_sim(arr_in):
	#function to create stokes parameter from simulations
	HH = arr_in[:,:,:,0]
	HV = arr_in[:,:,:,1]
	VH = arr_in[:,:,:,2]
	VV = arr_in[:,:,:,3]

	I = HH+VV
	Q = HH-VV
	U = HV+VH
	V = -1j*(HV-VH)
	P = np.sqrt(Q**2+U**2+V**2)

	new_arr = np.zeros((arr_in.shape[0],arr_in.shape[1],arr_in.shape[2],5),dtype=np.complex_)
	new_arr[:,:,:,0] = I
	new_arr[:,:,:,1] = Q
	new_arr[:,:,:,2] = U
	new_arr[:,:,:,3] = V
	new_arr[:,:,:,4] = P

	return new_arr

dirty_vis_stokes = create_stokes_from_sim(dirty_vis)
rfi_vis_stokes = create_stokes_from_sim(rfi)

##### converting complex values to absolute_mag
data_dirty_vis_stokes_mag = process_complex_arr(dirty_vis_stokes,output_format='absolute magnitude')
data_rfi_vis_stokes_mag = process_complex_arr(rfi_vis_stokes,output_format='absolute magnitude')

#### shuffling dataset to be able to train test etc 
#creating shuffle indices
sim_im_num = data_dirty_vis_stokes_mag.shape[0]
indices = np.arange(0,sim_im_num)
shuffle_indices = shuffle(indices, random_state=0)

#applying shuffling
data_dirty_vis_stokes_mag_shuffle = np.zeros(data_dirty_vis_stokes_mag.shape,float)
data_rfi_vis_stokes_mag_shuffle = np.zeros(data_rfi_vis_stokes_mag.shape,float)
for i in range(data_dirty_vis_stokes_mag_shuffle.shape[0]):
	data_dirty_vis_stokes_mag_shuffle[i,:,:] = data_dirty_vis_stokes_mag[shuffle_indices[i],:,:]
	data_rfi_vis_stokes_mag_shuffle[i,:,:] = data_rfi_vis_stokes_mag[shuffle_indices[i],:,:]


########## saving dataset for 2nd case 
save_h5file('shuf_seconddat_dirty_vis.h5',first_data_dirty_vis_mag_shuffle,dataset='shuf_seconddat_dirty_vis.h5')
save_h5file('shuf_seconddat_rfi_vis.h5',first_data_rfi_vis_mag_shuffle,dataset='shuf_seconddat_rfi_vis.h5')


########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################


#### loading the data
astro_sources = read_h5file('astro_sources.h5',dataset='astro_sources')
rfi = read_h5file('rfi.h5',dataset='rfi')

#creating noise and creating the dirty visibility
noise_array = create_noise(astro_sources,0,0.168)

################################## Data augmentation ##########################
np.random.seed(10)
rfi_intensity = np.random.uniform(0,3,10)
for i in range(10):
	print(i)
	if i == 0:
		rfi_augment = rfi_intensity[i]*rfi
		dirty_vis = astro_sources + noise_array + rfi_augment
	else:
		rfi_augment_1 = rfi_intensity[i]*rfi
		out = astro_sources + create_noise(astro_sources,0,0.168,seed=i) + rfi_augment_1
		dirty_vis = np.append(dirty_vis,out,axis=0)
		rfi_augment = np.append(rfi_augment,rfi_augment_1,axis=0)

# saving file as backup
save_h5file('aug_sample_dirty_vis.h5',dirty_vis,dataset='aug_sample_dirty_vis')
save_h5file('aug_sample_rfi_vis.h5',rfi_augment,dataset='aug_sample_rfi_vis')
del astro_sources
del noise_array
del rfi_augment_1
del out

###################################### creating 3rd dataset - baseline data cube ##########################
#looping over 4 polarization channels

size_arr = int((dirty_vis.shape[0]*4)/10)
third_data_dirty_vis = np.zeros((size_arr,10,100,4096),dtype=np.complex_)
third_data_rfi_vis = np.zeros((size_arr,10,100,4096),dtype=np.complex_)
k = 0
for i in range(dirty_vis.shape[0]-1):
	for j in range(4):
		third_data_dirty_vis[k,:,:,:] = dirty_vis[(i*10):((i+1)*10),:,:,j]
		third_data_rfi_vis[k,:,:,:] = rfi_augment[(i*10):((i+1)*10),:,:,j]
		k += 1

##### converting complex values to absolute_mag
third_data_dirty_vis_mag = process_complex_arr(third_data_dirty_vis,output_format='absolute magnitude')
third_data_rfi_vis_mag = process_complex_arr(third_data_rfi_vis,output_format='absolute magnitude')
del third_data_dirty_vis
del third_data_rfi_vis

#### shuffling dataset to be able to train test etc 
#creating shuffle indices
sim_im_num = third_data_dirty_vis_mag.shape[0]
indices = np.arange(0,sim_im_num)
shuffle_indices = shuffle(indices, random_state=0)

#applying shuffling
third_data_dirty_vis_mag_shuffle = np.zeros(third_data_dirty_vis_mag.shape,float)
third_data_rfi_vis_mag_shuffle = np.zeros(third_data_rfi_vis_mag.shape,float)

for i in range(third_data_dirty_vis_mag.shape[0]):
	third_data_dirty_vis_mag_shuffle[i,:,:,:] = third_data_dirty_vis_mag[shuffle_indices[i],:,:,:]
	third_data_rfi_vis_mag_shuffle[i,:,:,:] = third_data_rfi_vis_mag[shuffle_indices[i],:,:,:]

del third_data_dirty_vis_mag
del third_data_rfi_vis_mag

########## saving dataset for first case 
save_h5file('shuf_thirddat_dirty_vis.h5',third_data_dirty_vis_mag_shuffle,dataset='shuf_thirddat_dirty_vis')
save_h5file('shuf_thirddat_rfi_vis.h5',third_data_rfi_vis_mag_shuffle,dataset='shuf_thirddat_rfi_vis')


