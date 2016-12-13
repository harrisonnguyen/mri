import numpy as np
import nibabel as nib
import ml_cca

n_voxels = 91*109*91
T1_data = np.zeros((2,n_voxels))
fMRI_data = np.zeros((2,n_voxels))

in_file = "/project/RDS-SMS-NEUROIMG-RW/harrison/T1standardBL/SMART011T1_BL_brain_flirt.nii.gz"
img = nib.load(in_file)
data = img.get_data()
T1_data[0] = data.flatten()

in_file = "/project/RDS-SMS-NEUROIMG-RW/harrison/T1standardBL/SMART012T1_BL_brain_flirt.nii.gz"
img = nib.load(in_file)
data = img.get_data()
T1_data[1] = data.flatten()

T1_data.astype('float16')

in_file = "/project/RDS-SMS-NEUROIMG-RW/harrison/func2standardBL/SMART011rsfMRI_BL_brain_mcf_flirt.nii.gz"
img = nib.load(in_file)
data = img.get_data()
fMRI_data[0] = data[:,:,:,100].flatten()

in_file = "/project/RDS-SMS-NEUROIMG-RW/harrison/func2standardBL/SMART012rsfMRI_BL_brain_mcf_flirt.nii.gz"
img = nib.load(in_file)
data = img.get_data()
fMRI_data[1] = data[:,:,:,100].flatten()

fMRI_data.astype('float16')

cca = ml_cca.MaximumLikelihoodCCA(2)

cca.fit(fMRI_data,T1_data)

cca.transform()

#save the reuslts
np.save('/home/hngu4068/SMART/CCA_results/Cxx',cca.Cxx)
np.save('/home/hngu4068/SMART/CCA_results/Cxy',cca.Cxy)
np.save('/home/hngu4068/SMART/CCA_results/Cyx',cca.Cyx)
np.save('/home/hngu4068/SMART/CCA_results/Cyy',cca.Cyy)
np.save('/home/hngu4068/SMART/CCA_results/M1',cca.M1)
np.save('/home/hngu4068/SMART/CCA_results/M2',cca.M2)
np.save('/home/hngu4068/SMART/CCA_results/Pd',cca.Pd)
np.save('/home/hngu4068/SMART/CCA_results/W1',cca.W1)
np.save('/home/hngu4068/SMART/CCA_results/W2',cca.W2)
np.save('/home/hngu4068/SMART/CCA_results/Phi1',cca.Phi1)
np.save('/home/hngu4068/SMART/CCA_results/Phi2',cca.Phi2)
np.save('/home/hngu4068/SMART/CCA_results/mu1',cca.mu1)
np.save('/home/hngu4068/SMART/CCA_results/mu2',cca.mu2)
np.save('/home/hngu4068/SMART/CCA_results/E_z_x',cca.E_z_x)
np.save('/home/hngu4068/SMART/CCA_results/E_z_y',cca.E_z_y)
np.save('/home/hngu4068/SMART/CCA_results/var_z_x',cca.var_z_x)
np.save('/home/hngu4068/SMART/CCA_results/var_z_y',cca.var_z_y)
np.save('/home/hngu4068/SMART/CCA_results/E_z_xy',cca.E_z_xy)
np.save('/home/hngu4068/SMART/CCA_results/var_z_xy',cca.var_z_xy)




