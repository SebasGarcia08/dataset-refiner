from __future__ import division
import os
import logging
import shutil
import numpy as np
import sys

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

def countFiles(input_path):
    total_images = 0
    for _, _, filenames in os.walk(input_path):
        total_images += len(filenames)  
    return total_images

def write(msg):
    sys.stderr.write('\r{}'.format(msg))
