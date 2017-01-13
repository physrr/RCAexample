import os
import sys
import subprocess
import shutil
import numpy as np
from scipy import io as scio
import nibabel as nib
import RCA_functions
#import time

prog_help = "Script must be given (prep_ref_image_folder, main_image_FOLDER, target_lm, filenames_CONFIG, output_FOLDER\n"\
           "prep_ref_image_folder   = directory where reference images are listed (pre-prepared)\n"\
           "main_image_FOLDER       = directory containing the image, segmentation and landmarks to be tested\n"\
           "target_lm               = the landmarks of the first reference_image for registration\n"\
           "filenames_CONFIG        = config file giving the filename expected in main_image_FOLDER\n"\
           "output_FOLDER           = folder to output the files\n"

if len(sys.argv) < 6:
    sys.exit(prog_help)
            

if not os.path.isdir(os.path.abspath(str(sys.argv[1]))):
    msg = "Input is not a folder or doesn't exist: %s\n\n" % str(sys.argv[1])
    sys.exit(msg + prog_help)
else:
    prep_ref_img_FOLDER = os.path.abspath(str(sys.argv[1]))
                  
if not os.path.isdir(os.path.abspath(str(sys.argv[2]))):
    msg = "Input is not a folder or doesn't exist: %s\n\n" % str(sys.argv[2])
    sys.exit(msg + prog_help)
else:
    main_image_FOLDER = os.path.abspath(str(sys.argv[2]))
    main_image_NAME = os.path.split(main_image_FOLDER)[-1]
    
if not os.path.exists(os.path.abspath(str(sys.argv[3]))):
    msg = "Input is not a file or doesn't exist: %s\n\n" % str(sys.argv[3])
    sys.exit(msg + prog_help)
else:
    target_lm = os.path.abspath(str(sys.argv[3]))
    
if not os.path.exists(os.path.abspath(str(sys.argv[4]))):
    msg = "Input is not a file or doesn't exist: %s\n\n" % str(sys.argv[4])
    sys.exit(msg + prog_help)
else:
    filenames_CONFIG = os.path.abspath(str(sys.argv[4]))

if not os.path.exists(os.path.abspath(str(sys.argv[5]))):
    os.makedirs(sys.argv[5])
    root_FOLDER = os.path.abspath(str(sys.argv[5]))
else:
    root_FOLDER = os.path.abspath(str(sys.argv[5]))
    msg = "Output folder already exists: %s\n\n" % str(sys.argv[5])
    sys.stdout.write(msg)

prep=1
if len(sys.argv) == 7:
    if str(sys.argv[6])=="-noprep":
        prep=0
    else:
        msg = "Not a valid flag: %s\n\n" % str(sys.argv[6])
        sys.stderr.write(msg)
        
    
   

#SET USER DEFINED PARAMETERS  
runcmds = 1            # Run training and testing of RCA classifier (if 0 only evaluation is done)
target_num = 80          # Define the referece-image used as the target
offset = 20             # Offset of bounding-box/cropping from the landmarks of the target image
class_list = [0,1,2,4]  # Classes in the segmentations



#GATHER THE BIOBANKS IMAGE AND SEGMENTATION
execfile(filenames_CONFIG) # reads in "image_FILE, seg_FILE and landmarks_FILE"
main_image_FILE = os.path.abspath(os.path.join(main_image_FOLDER,image_FILE))
main_seg_FILE = os.path.abspath(os.path.join(main_image_FOLDER, seg_FILE))
main_landmarks_FILE = os.path.abspath(os.path.join(main_image_FOLDER,landmarks_FILE))

sys.stdout.write("\nRCA analysis on:\n"\
                 "Image:\t\t %s\nSegmentation:\t%s\n\n" % (main_image_FILE, main_seg_FILE))

sys.stdout.write("Preparing images...")

#CREATE FOLDERS FOR THE PREPARED REFERENCE IMAGES
prep_ref_seg_FOLDER = os.path.join(prep_ref_img_FOLDER, 'GTseg')
prep_ref_reg_FOLDER = os.path.join(prep_ref_img_FOLDER, "reg")
prep_ref_dof_FOLDER = os.path.join(prep_ref_reg_FOLDER, "dofs")

            
tmp = [f for f in sorted(os.listdir(prep_ref_img_FOLDER)) if os.path.isfile(os.path.join(prep_ref_img_FOLDER, f))]
reference_image_NAMES = [line.split('_')[1] for line in tmp]
reference_image_FILES = [os.path.abspath(os.path.join(prep_ref_img_FOLDER, f)) for f in sorted(os.listdir(prep_ref_img_FOLDER)) if os.path.isfile(os.path.join(prep_ref_img_FOLDER, f))]
reference_seg_FILES = [os.path.abspath(os.path.join(prep_ref_seg_FOLDER, f)) for f in sorted( os.listdir(prep_ref_seg_FOLDER)) if os.path.isfile(os.path.join(prep_ref_seg_FOLDER, f))]
 
output_FOLDER = os.path.join(root_FOLDER, main_image_NAME)


#create all the directories for RCA
rca_main_FOLDER = os.path.join(output_FOLDER, 'main_image')
rca_mainseg_FOLDER = os.path.join(rca_main_FOLDER, 'seg')
rca_FOLDER = os.path.join(output_FOLDER, 'RCA')
rca_train_PATH = os.path.join(rca_FOLDER, "RCA_classifier")
rca_test_PATH = os.path.join(rca_FOLDER, "test")
data_FOLDER = os.path.join(output_FOLDER, 'data')
rca_dof_FOLDER = os.path.join(rca_main_FOLDER, 'dof')
rca_maincropped_FOLDER = os.path.join(rca_main_FOLDER, 'cropped')
test_ref_FOLDER = os.path.join(rca_FOLDER,"test_ref")
test_seg_FOLDER = os.path.join(rca_FOLDER,"test_seg")
test_ref_cropped_FOLDER = os.path.join(test_ref_FOLDER,"cropped")
test_seg_cropped_FOLDER = os.path.join(test_seg_FOLDER,"cropped")

folders = [data_FOLDER, rca_main_FOLDER, rca_mainseg_FOLDER, rca_FOLDER, rca_train_PATH,rca_test_PATH, rca_dof_FOLDER,\
           rca_maincropped_FOLDER, test_ref_FOLDER, test_seg_FOLDER, test_ref_cropped_FOLDER, test_seg_cropped_FOLDER]
for fldr in folders:
    if not os.path.exists(fldr):
        os.makedirs(fldr)





        
cropped_mainimg_FILE = "cropped_%s.nii.gz" % main_image_NAME
cropped_mainseg_FILE = "croppedseg_%s.nii.gz" % main_image_NAME
cropped_mainimg_PATH = os.path.join(rca_maincropped_FOLDER, cropped_mainimg_FILE)
cropped_mainseg_PATH = os.path.join(rca_maincropped_FOLDER, cropped_mainseg_FILE)

points = RCA_functions.readVTK(main_landmarks_FILE)
centroid = list(RCA_functions.centeroidnp(points))
points = RCA_functions.readVTK(target_lm)    
dx, dy, dz = RCA_functions.boxsize(points)
dx, dy, dz= dx/1.8269, dy/1.8269, dz/10.0


if prep==0:
    shutil.copy(main_image_FILE, cropped_mainimg_PATH)
    shutil.copy(main_seg_FILE, cropped_mainseg_PATH)

    
if prep==1:
    #get the centorid of the landmarks of the biobank file around which to crop    
    cropped_mainimg_COMMAND = "mirtk extract-image-region %s %s -closest-patch %3.3f %3.3f %3.3f %3.0f %3.0f %3.0f  -margin %3.0f -pad 0" % (main_image_FILE, cropped_mainimg_PATH,\
                                                                                                                                            centroid[0], centroid[1], centroid[2],\
                                                                                                                                            dx, dy, dz, offset)
    cropped_mainseg_COMMAND = "mirtk extract-image-region %s %s -closest-patch %3.3f %3.3f %3.3f %3.0f %3.0f %3.0f  -margin %3.0f -pad 0" % (main_seg_FILE, cropped_mainseg_PATH,\
                                                                                                                                            centroid[0], centroid[1], centroid[2],\
                                                                                                                                            dx, dy, dz, offset)
    
    commands = [cropped_mainimg_COMMAND, cropped_mainseg_COMMAND]    
    if runcmds == 1:  
        for cmd in commands:
            with open(os.devnull, 'w') as fp: 
                proc = subprocess.Popen(cmd, shell=True, stdout=fp)
                out, err = proc.communicate()
                if proc.returncode != 0:
                    sys.stdout.write(err)


#register the uncropped ref images to the uncropped biobank 
for refnum, refimg in enumerate(reference_image_FILES):

    dof_FILE = "prreg_%s_to_%s.dof.gz" % (reference_image_NAMES[refnum], main_image_NAME)
    trans_landmarks = "trans_landmarks.vtk"
    regimg_FILE = "regimg_%s_to_%s.nii.gz" % (reference_image_NAMES[refnum], main_image_NAME)
    regseg_FILE = "regseg_%s_to_%s.nii.gz" % (reference_image_NAMES[refnum], main_image_NAME)
    
    ref_img_FILE = reference_image_FILES[refnum]
    ref_seg_FILE = reference_seg_FILES[refnum]
    dof_PATH= os.path.join(rca_dof_FOLDER, dof_FILE)
    trans_ref_FOLDER = os.path.join( test_ref_FOLDER, regimg_FILE)
    trans_seg_FOLDER = os.path.join( test_seg_FOLDER, regseg_FILE)
    
    
    cropped_img_FILE = "cropped_regimg_%s_to_%s.nii.gz" % (reference_image_NAMES[refnum], main_image_NAME)
    cropped_seg_FILE = "cropped_regseg_%s_to_%s.nii.gz" % (reference_image_NAMES[refnum], main_image_NAME)
    cropped_trans_ref_img = os.path.join(test_ref_FOLDER, "cropped", cropped_img_FILE)
    cropped_trans_ref_seg = os.path.join(test_seg_FOLDER, "cropped", cropped_seg_FILE)
    
    landmark_reg_COMMAND = "mirtk init-dof %s -rigid -disp %s %s" % (dof_PATH, main_landmarks_FILE, target_lm) 
    points_transform_COMMAND = "mirtk transform-points %s %s -dofin %s -target %s" % (target_lm, trans_landmarks, dof_PATH, main_image_FILE)
    transform_img_COMMAND = "mirtk transform-image %s %s -dofin %s -interp Linear -target %s" % (ref_img_FILE, trans_ref_FOLDER, dof_PATH, main_image_FILE)
    transform_seg_COMMAND = "mirtk transform-image %s %s -dofin %s -interp NN -target %s" % (ref_seg_FILE, trans_seg_FOLDER, dof_PATH, main_seg_FILE)
    
    cropped_mainimg_COMMAND = "mirtk extract-image-region %s %s -closest-patch %3.3f %3.3f %3.3f %3.0f %3.0f %3.0f  -margin %3.0f -pad 0" % (trans_ref_FOLDER, cropped_trans_ref_img,\
                                                                                                                                    centroid[0], centroid[1], centroid[2],\
                                                                                                                                    dx, dy, dz, offset)
    cropped_mainseg_COMMAND = "mirtk extract-image-region %s %s -closest-patch %3.3f %3.3f %3.3f %3.0f %3.0f %3.0f  -margin %3.0f -pad 0" % (trans_seg_FOLDER, cropped_trans_ref_seg,\
                                                                                                                                    centroid[0], centroid[1], centroid[2],\
                                                                                                                                    dx, dy, dz, offset)
   
    #compute similairity to test-image
    
       
    commands = [landmark_reg_COMMAND, points_transform_COMMAND, transform_img_COMMAND, transform_seg_COMMAND, cropped_mainimg_COMMAND, cropped_mainseg_COMMAND]

    if runcmds == 1:            
        for cmd in commands:
            with open(os.devnull, 'w') as fp: 
                proc = subprocess.Popen(cmd, shell=True, stdout=fp)
                proc.wait()
                out, err = proc.communicate()
                if proc.returncode != 0:
                    print(err)       
        
    
test_ref_FILE = [os.path.join(test_ref_cropped_FOLDER,f) for f in os.listdir(test_ref_cropped_FOLDER) if os.path.isfile(os.path.join(test_ref_cropped_FOLDER, f))]
test_ref_NAME = [f for f in os.listdir(test_ref_cropped_FOLDER) if os.path.isfile(os.path.join(test_ref_cropped_FOLDER, f))]    
test_seg_FILE = [os.path.join(test_seg_cropped_FOLDER,f) for f in os.listdir(test_seg_cropped_FOLDER) if os.path.isfile(os.path.join(test_seg_cropped_FOLDER, f))]
test_seg_NAME = [f for f in os.listdir(test_seg_cropped_FOLDER) if os.path.isfile(os.path.join(test_seg_cropped_FOLDER, f))]





                          
sys.stdout.write("done!\n")      
sys.stdout.write("Performing RCA...")



for imgnum, im2reg in enumerate(sorted(test_ref_FILE)):

    commands = []    

    ref_name = os.path.basename(os.path.abspath(im2reg))
    filename_base = "%s_to_%s" % (main_image_NAME, reference_image_NAMES[imgnum])
    imgfilename = "warped_img_%s.nii.gz" % filename_base
    segfilename = "warped_seg_%s.nii.gz" % filename_base
    image = os.path.join(im2reg)
    regfolder = os.path.abspath("%s_reg" % main_image_NAME)
    warped_folder = os.path.join(rca_test_PATH, "warped_imgs")
    imgoutputfile = os.path.join(warped_folder, imgfilename)
    segoutputfile = os.path.join(rca_test_PATH, segfilename)
    
    commands.append( "dropreg -s %s -t %s -o %s -c "\
                    #"-l --lsim 1 --ltype 2 --lsampling 0.1 --llevels 4 4 1 2 2 1 "\
                    "-n --nsim 1 --nffd 40 --nlambda 2 --nlevels 4 4 1 2 2 1" \
                    % (cropped_mainimg_PATH, image, imgoutputfile)
                )

    transform_txt = os.path.join(warped_folder, "warped_img_%s_transform.txt" % filename_base)
    xf = os.path.join(warped_folder, "warped_img_%s_field_x.nii.gz" % filename_base)
    yf = os.path.join(warped_folder, "warped_img_%s_field_y.nii.gz" % filename_base)
    zf = os.path.join(warped_folder, "warped_img_%s_field_z.nii.gz" % filename_base)
    
    commands.append("dropreg -s %s -t %s -o %s --ointerp 0 "\
                        "--transform %s --fx %s --fy %s --fz %s"\
                        % (cropped_mainseg_PATH, image, segoutputfile, transform_txt, xf, yf, zf)
                        )
    
    clean_up = "rm -rf %s %s %s %s" % (xf, yf, zf, transform_txt)
    commands.append(clean_up)
 
    if runcmds==1:          
        for cmd in commands:
            with open(os.devnull, 'w') as fp: 
                proc = subprocess.Popen(cmd, shell=True, stdout=fp)
                out, err = proc.communicate()
                if proc.returncode != 0:
                    sys.stdout.write(err)


                                      
#get all of the results from RCA
#get all of the GT refrence segmentations          
rca_results_PATHS = [os.path.join(rca_test_PATH,f) for f in sorted(os.listdir(rca_test_PATH)) if os.path.isfile(os.path.join(rca_test_PATH, f))]
rca_croppedseg_PATHS = [os.path.join(test_seg_cropped_FOLDER,f) for f in sorted(os.listdir(test_seg_cropped_FOLDER)) if os.path.isfile(os.path.join(test_seg_cropped_FOLDER, f))]
           
         
         
         
         
         
sys.stdout.write("done!\n")                     
sys.stdout.write("Evaluating")


curr_RCA_metrics_FILE = os.path.join(data_FOLDER, main_image_NAME)
RCA_metrics = []
DSC = dict()
commands = []

DSC['ImageID'] = main_image_NAME
DSC['Classes'] =  class_list
for imgnum, path in enumerate(rca_results_PATHS):
    
    if imgnum % 20 == 0:
        sys.stdout.write('.')
        
    #load each image
    GT_seg = nib.load(rca_results_PATHS[imgnum]).get_data()
    RCA_seg= nib.load(rca_croppedseg_PATHS[imgnum]).get_data()
    
    #get metrics for each class then for the average
    currDSC = []

    
    for cls in class_list:
        #DSC        
        currDSC.append(RCA_functions.dice(np.ravel(GT_seg==cls), np.ravel(RCA_seg==cls))) #each class

    currDSC.append(RCA_functions.dice(np.ravel(GT_seg>0), np.ravel(RCA_seg>0))) #average

    DSC["a%s" % reference_image_NAMES[imgnum]] = np.asarray(currDSC)
 
#GET HIGHEST DSC OUT OF THE REFERENCE IMAGES PER CLASS
class_max = []
class_maxDSC = []

 
for i in range(0,5):
    x= []
    for key in DSC:
        if key == 'Classes' or key == 'ImageID': continue
    
        x.append(DSC[key][i])
    class_maxDSC.append([np.min(x), np.argmin(x)+1]) #+1 because of 0-indexing (should be the atlas number)
    #class_maxDSC.append([np.sort(x)[-3:-1].mean(), np.argsort(x)[-3]])

DSC['MaxDSC'] = np.asarray(class_maxDSC)

#create a .mat file for matlab
scio.savemat(curr_RCA_metrics_FILE, DSC)
sys.stdout.write("done!\n\n")

sys.stdout.write("Predicted DSC for %s = %2.3f\n\n" %(main_image_NAME, DSC['MaxDSC'][4][0]))

sys.exit(0)