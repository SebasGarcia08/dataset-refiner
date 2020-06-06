from setuptools import setup

setup(
    name = "dataset_refiner",
    version = "0.0.1",
    author="Sebastián García Acosta",
    description = "Code written in order to make Nando to be a professional data refiner",
    packages = [],
    install_requires=[
      "mxnet == 1.5.0",
      "insightface",
      "numpy",
      "opencv-python",
      "tqdm"  
    ],
    
    zip_safe = False
)