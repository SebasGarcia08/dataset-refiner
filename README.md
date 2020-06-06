#Installation

```
git clone https://github.com/SebasGarcia08/dataset-refiner
cd dataset-refiner
pip install .
```

# Usage

```
usage: cleaner.py [-h] -indir INPUT_BASE_DIR -out OUTPUT_PATH [-crop] [-flat]
                  [-duplicate]

Script for detecting faces in a given folder and its subdirectories

optional arguments:
  -h, --help            show this help message and exit
  -indir INPUT_BASE_DIR, --input-base-dir INPUT_BASE_DIR
                        Path to the directory where images or folders of
                        images are
  -out OUTPUT_PATH, --output-path OUTPUT_PATH
                        Path of the folder where faces images will be saved
  -crop, --crop-faces   Crop faces detected in images and save each one
  -flat, --same-out-dir
                        Whether to save all images in dirctory specified in
                        -out --output-path and not imitate directory structure
                        from the path specified in -indir --input-base-dir
  -duplicate, --duplicate-faces
                        Whether to save the original images of the extracted
                        faces also. Only valid if -crop --crop-faces is pass
                        as argument
```

## Windows

* **Open a terminal as administrator**

* For saving images and its faces:

```
python cleaner.py -indir <input_path> -out <output_path> -crop -duplicate
```

## Running with a Anaconda Environment and in linux

```
sudo env "PATH=$PATH" python cleaner.py -indir <input_path> -out <output_path>
```


