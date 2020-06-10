from argparse import ArgumentParser
import insightface
import cv2
import os
import logging
import traceback
import shutil
from tqdm import tqdm, tqdm_gui
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from refiner.FaceFilterer import FaceFilterer
import multiprocessing
from multiprocessing import Pool
from p_tqdm import p_map
import sys

def main(args):
    logging.info(' Reading files in {}'.format(args["INPUT_BASE_DIR"]))
    if not args["save_in_same_output_folder"]:
        copyDirectoryStructure(args["INPUT_BASE_DIR"], args["OUTPUT_PATH"])
    else:
        if os.path.exists(args["OUTPUT_PATH"]):
            logging.error(" {} path already exists".format(args["OUTPUT_PATH"]))
            return 
        else:            
            os.mkdir(args["OUTPUT_PATH"])
    try:
        if os.path.exists(args["INPUT_BASE_DIR"]):
            num_imgs = 0
            num_faces_detected = 0
            num_images_filtered = 0
            num_masked_faces_detected = 0
            
            for dirpath, _, filenames in os.walk(args["INPUT_BASE_DIR"]):
                for filename in filenames:
                    num_imgs += 1
            
            pbar = tqdm(total=num_imgs, desc="Progress", unit="Images")
            pbar.write(f"{num_imgs} images found")
            
            for img_number, (dirpath, _, filenames) in enumerate(os.walk(args["INPUT_BASE_DIR"])):
                for filename in filenames:
                    try:
                        partialPath = os.path.sep.join([ dirpath[ len(args["INPUT_BASE_DIR"]): ], filename ])
                        src = os.path.sep.join([args["INPUT_BASE_DIR"], partialPath])
                        img = cv2.imread(src)
                        img = cv2.resize(img, (255, 255))
                        bbox, _ = model.detect(img, threshold=0.5, scale=1.0)
                        
                        if len(bbox) > 0:
                            num_images_filtered += 1
                            if args["save_in_same_output_folder"]:
                                out = os.path.sep.join([args["OUTPUT_PATH"], filename])
                            else:
                                out = os.path.sep.join([args["OUTPUT_PATH"], partialPath])      
                            
                            if args["keep_only_imgs_with_faces"]:
                                if args["move_images"]:
                                    shutil.move(src, out)
                                else:
                                    cv2.imwrite(out, img)
                                
                            elif args["crop_faces"] or args["keep_only_imgs_with_masked_faces"]:
                                
                                current_num_faces_detected = 0                                                                
                                faces = []
                                croppedImages = []
                                
                                for box in bbox: 
                                    try:
                                        if args["keep_only_imgs_with_masked_faces"]:
                                            x,y,w,h,_ = list(map(int, box))
                                            imgCrop = img[y:y+h,x:x+w]
                                            croppedImages.append(imgCrop)
                                            face = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2RGB)
                                            face = cv2.resize(face, (224, 224))
                                            face = img_to_array(face)
                                            face = preprocess_input(face)
                                            face = np.expand_dims(face, axis=0)
                                            faces.append(face)
                                    except Exception as e:
                                        # logging.error(traceback.format_exc()) 
                                        pbar.write(traceback.format_exc())        
                                
                                saveImg = True
                                if len(faces) > 0:
                                    preds = maskNet.predict(faces)
                                    for i, (imgCrop, pred) in enumerate(zip(croppedImages, preds)):
                                        out = out.replace(".jpg","")
                                        out += f"face_No{str(i+1)}.jpg"
                                        saveCroppedImg = True                                        
                                        if args["crop_faces"]:                                            
                                            if args["keep_only_imgs_with_masked_faces"]:
                                                pMask, pNotMask = np.squeeze(pred)
                                                saveCroppedImg = pMask > .3

                                            if saveCroppedImg:
                                                try:
                                                    imgCrop = cv2.resize(imgCrop, (224,224)) #this resizing could rise exception
                                                    cv2.imwrite(out, imgCrop)
                                                    num_faces_detected += 1
                                                    current_num_faces_detected += 1   
                                                except:
                                                    try:
                                                        cv2.imwrite(out, imgCrop) # if so, then save images as is, iwithout resizing
                                                        num_faces_detected += 1   
                                                        current_num_faces_detected += 1   
                                                    except Exception as e:
                                                        pbar.write(str(e))
                                            else:
                                                saveImg = False
                                        
                                    if args["duplicate_img_of_faces"]:
                                        if args["keep_only_imgs_with_masked_faces"]:
                                            if len(faces) == 1 and saveImg: 
                                                if args["move_images"]:
                                                    shutil.move(src, out)
                                                else:
                                                    cv2.imwrite(out, img)
                                        else:
                                            if args["move_images"]:
                                                    shutil.move(src, out)
                                            else:
                                                cv2.imwrite(out, img)                                                                                        
                            
                            if args["crop_faces"]:
                                s = " masked" if args["keep_only_imgs_with_masked_faces"] else " "
                                msg = f"Detected{s} faces: {current_num_faces_detected} - Total: {num_faces_detected} - Percentage of faces over images: {(num_faces_detected/(img_number+1))*100}%"
                                pbar.write(msg)
                            else:
                                pbar.write(f"Filtered images: {num_images_filtered} - Percemtage of saved images: {(num_images_filtered/img_number)*100}%")
                        pbar.update(1)
                    except Exception as e:  
                        pbar.write(str(e))
        else:
            raise FileNotFoundError("Path does not exists")
        
    except Exception as e:
        logging.log(40, traceback.format_exc())

def yieldPaths(input_path, output_path, flat=False): 
    for dirpath, _, filenames in os.walk(input_path):
        for filename in filenames:
            partialPath = os.path.sep.join([ dirpath[ len(input_path): ], filename])
            src = os.path.sep.join([input_path, partialPath])
            if flat:
                out = os.path.sep.join([output_path, filename])
            else:   
                out = os.path.sep.join([output_path, partialPath])      
            yield (src, out)

def copyDirectoryStructure(base_path, output_path):
    if os.path.exists(base_path):
        res = "yes"
        for dirpath, _ , _ in os.walk(base_path):
            structure = os.path.sep.join([ output_path, dirpath[ len(base_path): ] ])
            try:
                logging.info(" Creating {} path".format(structure))
                if res == "yesAll":
                    os.makedirs(structure, exist_ok=True)
                else:
                    os.mkdir(structure)
            except FileExistsError:
                msg = "Path {} already exists, do you want to overwrite it? [yes/no/yesAll/noAll]: ".format(structure)
                res = input(msg)
            if res == "noAll":
                break
            if res != "yes" and res != "no" and res != "yesAll" and res != "noAll":
                print("Invalid choice")
                break         
    else:
        logging.error("File does not exists")

def copyFile(src, dst, buffer_size=10485760, perserveFileDate=True):
    '''
    From: https://blogs.blumetech.com/blumetechs-tech-blog/2011/05/faster-python-file-copy.html
    Copies a file to a new location. Much faster performance than Apache Commons due to use of larger buffer
    @param src:    Source File
    @param dst:    Destination File (not file path)
    @param buffer_size:    Buffer size to use during copy
    @param perserveFileDate:    Preserve the original file date
    '''
    #    Check to make sure destination directory exists. If it doesn't create the directory
    dstParent, dstFileName = os.path.split(dst)
    if(not(os.path.exists(dstParent))):
        os.makedirs(dstParent)

    #    Optimize the buffer for small files
    buffer_size = min(buffer_size,os.path.getsize(src))
    if(buffer_size == 0):
        buffer_size = 1024

    if shutil._samefile(src, dst):
        raise shutil.Error("`%s` and `%s` are the same file" % (src, dst))
    for fn in [src, dst]:
        try:
            st = os.stat(fn)
        except OSError:
            # File most likely does not exist
            pass
        else:
            # XXX What about other special files? (sockets, devices...)
            if shutil.stat.S_ISFIFO(st.st_mode):
                raise shutil.SpecialFileError("`%s` is a named pipe" % fn)
    with open(src, 'rb') as fsrc:
        with open(dst, 'wb') as fdst:
            shutil.copyfileobj(fsrc, fdst, buffer_size)

    if(perserveFileDate):
        shutil.copystat(src, dst)

def countImages(input_path):
    total_images = 0
    for _, _, filenames in os.walk(input_path):
        total_images += len(filenames)  
    return total_images

def write(msg):
    sys.stderr.write('\r{}'.format(msg))

def run(src, out):
    try:
        img = cv2.imread(src)
        img = cv2.resize(img, (255, 255))
        bbox, _ = self.model.detect(img, threshold=0.5, scale=1.0)
        
        if len(bbox) > 0:
            if move_files:
                shutil.move(src, out)
            else:
                self.copyFile(src, out)
            num_filtered_images += 1
            ratio = round((self.num_filtered_images / (img_number + 1)) * 100, 3)
            self.write("Filtered imgs: {}| % Imgs saved: {}".format( self.num_filtered_images, ratio))
    except Exception as e:  
        self.write(str(e))

def filterFace():
    pass
        
if __name__ == '__main__':
    # Initialize parser
    parser = ArgumentParser(
        description="Script for detecting faces in a given folder and its subdirectories"
    )
    
    parser.add_argument("-in", "--input-path", 
                        type=str,
                        required=True,
                        dest="INPUT_BASE_DIR", 
                        help="Path to the directory where images or folders of images are\n")
    
    parser.add_argument("-out","--output-path",
                        type=str, 
                        required=True,
                        dest = "OUTPUT_PATH", 
                        help="Path of the folder where faces images will be saved\n")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--keep-faces",
                        action="store_true",
                        dest="keep_only_imgs_with_faces",
                        help = "Set the keeping criteria to images with faces. Whether to keep images stored from [-out, --output-path] to [-in, --input-path] only if contain faces")

    group.add_argument("--keep-faces-with-mask",
                        action="store_true",
                        dest="keep_only_imgs_with_masked_faces",
                        help = "Set the keeping criteria to images with faces that wear mask. Whether to keep images stored from [-out, --output-path] to [-in, --input-path] only if contain faces with mask")
    
    parser.add_argument("-move", "--move-kept-images", 
                        action="store_true",
                        default=False,
                        dest = "move_images",
                        help = "Whether to move kept images from [-in, --input-path] to [-out, --output-path] in such a way that in the remaining images in [-in --input-path] are the ones that did not apply the criteria.")

    parser.add_argument("-crop","--crop-faces", 
                        action='store_true',
                        dest="crop_faces",
                        default=False, 
                        help="Crop faces detected in images and save each one\n")
    
    parser.add_argument("-flat", "--same-out-dir",
                        action='store_true',
                        dest="save_in_same_output_folder",
                        default=False,
                        help="Whether to save all images in dirctory specified in -out --output-path and not imitate directory structure from the path specified in -indir --input-base-dir\n")

    parser.add_argument("-duplicate", "--duplicate-img-faces",
                        action="store_true", 
                        dest="duplicate_img_of_faces",
                        default=False, 
                        help="Whether to save the original images of the extracted faces also. Only valid if -crop --crop-faces is passed as argument")
    
    parser.add_argument("-model", "--classification-model", 
                        type=str,
                        dest = "classification_model", 
                        default="resources/model_with_1400_masked_samples.h5")

    kwargs = vars(parser.parse_args())
    logging.basicConfig(level=logging.INFO)
    logging.info(" Preparing model...")
    
    model = insightface.model_zoo.get_model('retinaface_r50_v1')
    model.prepare(ctx_id = -1, nms=0.4)
    
    if kwargs["keep_only_imgs_with_masked_faces"]:
        logging.info(" Loading classification model...")
        maskNet = tf.keras.models.load_model(kwargs["classification_model"], compile=False)
    
    main2(kwargs)    