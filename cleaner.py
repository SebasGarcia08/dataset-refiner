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
            
            for dirpath, dirnames, filenames in os.walk(args["INPUT_BASE_DIR"]):
                for filename in filenames:
                    num_imgs += 1
            
            print(num_imgs)
            pbar = tqdm(total=num_imgs, desc="Progress", unit="Images")
            
            for dirpath, _, filenames in os.walk(args["INPUT_BASE_DIR"]):
                for filename in filenames:
                    partialPath = os.path.sep.join([ dirpath[ len(args["INPUT_BASE_DIR"]): ], filename ])
                    src = os.path.sep.join([args["INPUT_BASE_DIR"], partialPath])
                    img = cv2.imread(src)
                    img = cv2.resize(img, (255, 255))
                    bbox, _ = model.detect(img, threshold=0.5, scale=1.0)
                    
                    if args["save_in_same_output_folder"]:
                        out = os.path.sep.join([args["OUTPUT_PATH"], filename])
                    else:
                        out = os.path.sep.join([args["OUTPUT_PATH"], partialPath])                                            
                    if args["crop_faces"]:
                        for i, box in enumerate(bbox): 
                            out = out.replace(".jpg","")
                            out += f"{str(i+1)}.jpg"
                            try:
                                x,y,w,h,_ = list(map(int, box))
                                imgCrop = img[y:y+h,x:x+w]
                                try:
                                    imgCrop = cv2.resize(imgCrop, (255,255))
                                    cv2.imwrite(out, imgCrop)
                                    num_faces_detected += 1   
                                except:
                                    try:
                                        cv2.imwrite(out, imgCrop)
                                        num_faces_detected += 1   
                                    except:
                                        pass
                            except Exception as e:
                                # logging.error(traceback.format_exc()) 
                                pbar.write(str(e))
                    else:
                        if len(bbox) > 0:
                            cv2.imwrite(out, img)
                            num_images_filtered += 1
                    if args["crop_faces"]:
                        pbar.write(f"Detected faces: {len(bbox)} - Total: {num_faces_detected} - Percentage of faces over images: / {(num_faces_detected/num_imgs)*100}%")
                    else:
                        pbar.write(f"Images filtered: {num_images_filtered} - Percemtage of images saved: {(num_images_filtered/num_imgs)*100}%")
                    pbar.update(1)

        else:
            raise FileNotFoundError("Path does not exists")
        
    except Exception as e:
        logging.log(40, traceback.format_exc())
            
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
    
    parser.add_argument("-indir", "--input-base-dir", 
                        type=str,
                        required=True,
                        dest="INPUT_BASE_DIR", 
                        help="Path to the directory where images or folders of images are\n")
    
    parser.add_argument("-out","--output-path",
                        type=str, 
                        required=True,
                        dest = "OUTPUT_PATH", 
                        help="Path of the folder where faces images will be saved\n")
    
    parser.add_argument("-crop","--crop-faces", 
                        action='store_true',
                        dest="crop_faces",
                        default=False, 
                        help="Crop faces detected in images and save each one\n")
    
    parser.add_argument("-flat", "--same-out-dir",
                        action='store_true',
                        dest="save_in_same_output_folder",
                        default=False,
                        help="Wheter to save all images in dirctory specified in -out --output-path and not \
                              imitate directory structure from the path specified in -indir --input-base-dir\n")
    
    kwargs = vars(parser.parse_args())
    logging.info("Preparing model...")
    model = insightface.model_zoo.get_model('retinaface_r50_v1')
    model.prepare(ctx_id = -1, nms=0.4)
    
    logging.basicConfig(level=logging.INFO)

    main(kwargs)    