#Installation

```
git clone https://github.com/SebasGarcia08/face-extractor
cd face-extractor
pip install .
cd ..
```

# Usage

```
usage: cleaner.py [-h] -in INPUT_BASE_DIR -out OUTPUT_PATH [-crop] [-flat]
                  [-duplicate]

Script for detecting faces in a given folder and its subdirectories

optional arguments:
  -h, --help            show this help message and exit
  -in INPUT_BASE_DIR, --input-path INPUT_BASE_DIR
                        Path to the directory where images or folders of
                        images are
  -out OUTPUT_PATH, --output-path OUTPUT_PATH
                        Path of the folder where faces images will be saved
  -crop, --crop-faces   Crop faces detected in images and save each one
  -flat, --same-out-dir
                        Whether to save all images in dirctory specified in
                        -out --output-path and not imitate directory structure
                        from the path specified in -indir --input-base-dir
  -duplicate, --duplicate-img-faces
                        Whether to save the original images of the extracted
                        faces also. Only valid if -crop --crop-faces is passed
                        as argument
```

## Windows

* **Open a terminal as administrator**

* For saving images and its faces:

```
python face-extractor/cleaner.py --input-path <input_path> --output-path <output_path> \
                                 --crop-faces --duplicate-img-faces
```

## Running with a Anaconda Environment and in linux

```
sudo env "PATH=$PATH" python cleaner.py -indir <input_path> -out <output_path>
```


