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
                                
                            if args["crop_faces"] or args["keep_only_imgs_with_masked_faces"]:
                                current_num_faces_detected = 0                                                                
                                
                                for i, box in enumerate(bbox): 
                                    out = out.replace(".jpg","")
                                    out += f"face_No{str(i+1)}.jpg"
                                    try:
                                        x,y,w,h,_ = list(map(int, box))
                                        imgCrop = img[y:y+h,x:x+w]
                                        
                                        saveImg = True
                                        if args["keep_only_imgs_with_masked_faces"]:
                                            face = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2RGB)
                                            face = cv2.resize(face, (224, 224))
                                            face = img_to_array(face)
                                            face = preprocess_input(face)
                                            face = np.expand_dims(face, axis=0)
                                            (pMask, pNotMask) = np.squeeze(maskNet.predict(face))    
                                            saveImg = pMask > .6 and pNotMask < .5
                                        
                                        if saveImg and args["crop_faces"]:
                                            try:
                                                imgCrop = cv2.resize(imgCrop, (255,255)) #this resizing could rise exception
                                                cv2.imwrite(out, imgCrop)
                                                num_faces_detected += 1
                                                current_num_faces_detected += 1   
                                            except:
                                                try:
                                                    cv2.imwrite(out, imgCrop) # if so, then save images as is, iwithout resizing
                                                    num_faces_detected += 1   
                                                    current_num_faces_detected += 1   
                                                except:
                                                    pass
                                    except Exception as e:
                                        # logging.error(traceback.format_exc()) 
                                        pbar.write(traceback.format_exc())
                                
                                if args["duplicate_img_of_faces"] and len(bbox) == 1 and saveImg: 
                                    if args["move_images"]:
                                        shutil.move(src, out)
                                    else:
                                        cv2.imwrite(out, img)
                            
                            if args["crop_faces"]:
                                msg = f"Detected faces: {current_num_faces_detected} - Total: {num_faces_detected} - Percentage of faces over images: {(num_faces_detected/(img_number+1))*100}%"
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
    group.add_argument("--keep-faces-imgs",
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
    
    logging.info(" Loading classification model...")
    maskNet = tf.keras.models.load_model(kwargs["classification_model"], compile=False)
    
    main(kwargs)    