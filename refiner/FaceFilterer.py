import os 
import cv2
import insightface
import logging
import sys
import shutil

class FaceFilterer:
    
    def __init__(self, input_path, output_path, face_detection_model, flat=False, move=False):
        logging.basicConfig(level=logging.INFO)        
        self.input_path = input_path
        self.output_path = output_path
        self.save_in_same_output_dir = flat
        self.move_files = move
        self.model = face_detection_model
        self.total_images = 0
        self.num_filtered_images = 0
    
    def prepare(self):
        self.countImages()
        if not self.save_in_same_output_dir:
            self.createDestinationFolders()
        
    def countImages(self):
        for _, _, filenames in os.walk(self.input_path):
            self.total_images += len(filenames)

    def createDestinationFolders(self):
        base_path = self.input_path
        output_path = self.output_path
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
            
    def run(self):
        for img_number, (dirpath, _, filenames) in enumerate(os.walk(self.input_path)):
            for filename in filenames:
                try:
                    partialPath = os.path.sep.join([ dirpath[ len(self.input_path ): ], filename ])
                    src = os.path.sep.join([self.input_path, partialPath])
                    img = cv2.imread(src)
                    img = cv2.resize(img, (255, 255))
                    bbox, _ = self.model.detect(img, threshold=0.5, scale=1.0)
                    
                    if len(bbox) > 0:
                        if self.save_in_same_output_dir:
                            out = os.path.sep.join([self.output_path, filename])
                        else:
                            out = os.path.sep.join([self.output_path, partialPath])      
                        
                        if self.move_files:
                            shutil.move(src, out)
                        else:
                            self.copyFile(src, out)
                        self.num_filtered_images += 1
                        ratio = round((self.num_filtered_images / (img_number + 1)) * 100, 3)
                        self.write("Filtered imgs: {}| % Imgs saved: {}".format( self.num_filtered_images, ratio))
                except Exception as e:  
                    self.write(str(e))
                    
    @staticmethod 
    def write(msg):
        sys.stderr.write('\r{}'.format(msg))

    @staticmethod
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