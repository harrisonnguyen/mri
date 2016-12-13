import numpy as np
from nipype import Node, Workflow
from nipype.interfaces import fsl
import nipype.pipeline.engine as pe

standard_brain = "/project/RDS-SMS-NEUROIMG-RW/harrison/MNI152_T1_2mm_brain.nii.gz"

#standard_brain = "/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm_brain.nii.gz"

motion_correction = pe.MapNode(fsl.MCFLIRT(dof=6), name="motion_correction",iterfield = ['in_file'])
fmri_skullstrip = pe.MapNode(fsl.BET(mask=True,functional=True), name="fmri_skullstrip",iterfield = ['in_file'])
structural_skullstrip = pe.MapNode(fsl.BET(mask=True), name="structural_skullstrip",iterfield = ['in_file'])
epi_registration = pe.MapNode(fsl.epi.EpiReg(echospacing =0.0007 ),name = "epi_registration",iterfield = ['epi','t1_brain','t1_head'])
highres2standard = pe.MapNode(fsl.FLIRT(reference = standard_brain,cost = 'corratio',dof=6,interp = 'trilinear',searchr_x = [-90,90],searchr_y=[-90,90],searchr_z = [-90,90]),name = 'highres2standard',iterfield = ['in_file'])

standard2highres = pe.MapNode(fsl.ConvertXFM(invert_xfm=True),name = "standard2highres",iterfield = ['in_file'])

example_func2standard = pe.MapNode(fsl.ConvertXFM(concat_xfm=True),name = "example_func2standard",iterfield = ['in_file','in_file2'])

example_func2standard_full = pe.MapNode(fsl.FLIRT(interp='trilinear',apply_xfm = True,echospacing = 0.0007,reference=standard_brain),name = "example_func2standard_full",iterfield = ['in_file','in_matrix_file'])

wf = Workflow(name="prelim")  # Workflows need names too

wf.connect([(fmri_skullstrip,motion_correction,[('out_file','in_file')]),
            (structural_skullstrip,epi_registration,[('out_file','t1_brain')]),
            (motion_correction,epi_registration,[('out_file','epi')]),
            (epi_registration,example_func2standard,[('epi2str_mat','in_file')]),
            (structural_skullstrip,highres2standard,[('out_file','in_file')]),
            (highres2standard,example_func2standard,[('out_matrix_file','in_file2')]),
            (motion_correction,example_func2standard_full,[('out_file','in_file')]),
            (example_func2standard,example_func2standard_full,[('out_file','in_matrix_file')])])

fmri_path = '/project/RDS-SMS-NEUROIMG-RW/harrison/rsfMRI_dcm2nii_4D/'
#fmri_path = '/RDSMount/STUDY_DATA/SMART_DATA/HARRISON_WORK/rsfMRI_dcm2nii_4D/'
fmri_files = [fmri_path +'SMART001rsfMRI_BL.nii', fmri_path +'SMART002rsfMRI_BL.nii']
structural_path = '/project/RDS-SMS-NEUROIMG-RW/harrison/T1_nii_dcm2nii/'
#structural_path = '/RDSMount/STUDY_DATA/SMART_DATA/HARRISON_WORK/T1_nii_dcm2nii/'
structural_files = [structural_path + 'SMART001T1_BL.nii',structural_path + 'SMART002T1_BL.nii']
fmri_skullstrip.inputs.in_file = fmri_files
structural_skullstrip.inputs.in_file = structural_files
epi_registration.inputs.t1_head = structural_files

wf.base_dir = "/scratch/RDS-SMS-NEUROIMG-RW/harrison/working_dir"
wf.run(plugin="MultiProc", plugin_args={"n_proc": 2})
#wf.run()
