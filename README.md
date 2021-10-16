# Artificially Intelligent Food Assistant for the Visually Impaired

Our system is an interface which utilizes many advancements in Artificial Intelligence to make visual evaluations of a variety of both fresh and packaged food for the visually impaired.

## Dependencies

You will first need to install the anaconda package manager.\
You can follow the instructions by clicking the [Anaconda Installing Guide](https://docs.anaconda.com/anaconda/install/index.html)\
Optionally, You can follow [Miniconda Installing Guide](https://docs.conda.io/en/latest/miniconda.html)\
Any other package manager will work only if it can install all the packages in our requirements files

Once you have a package manager, It is suggested to just run the inference scripts as delineated by the instructions below, and whenever
you have an ImportError: No module named ..., then you install the package using your package manager.\
**For anaconda, every package other than the effdet package was installed through anaconda. Install the effdet package using PyPI over [here](https://pypi.org/project/effdet/)**

You can install install packages by running:
```>> conda install [library]
```
It is recommended to first try downloading the package as shown above. If the shell tells you the package does not exist then download the package from the conda forge channel as shown below
```>> conda install -c conda-forge [library]
```
You can find more about anaconda by looking at their [documentation](https://docs.anaconda.com/)

## Git Cloning the project

To git clone the project run the following in your shell ```>> git clone https://github.com/SarthakJaingit/Artificially-Intelligent-Food-Assistant-for-the-Visually-Impaired.git```\
A further reference you can check out if you need assistance is over [here](https://git-scm.com/book/en/v2/Git-Basics-Getting-a-Git-Repository)\
If you have trouble git cloning the project, then another option will be to press the repo link and download the code via zip file by toggling the green box on the top right that says **Code**\
If you download by zip file, make sure to uncompress the zip to see the actual code.

## Running the Script onto a Shell

After git cloning this repo and installing all packages with anaconda, you will have the ability to run our programs into your shell.\
Please refer to the following example implementation and docs
```>> python3 inference_device.py apple-5265125_1280.jpg  --device cpu --model_name mobilenet_fasterrcnn --confidence_thresh 0.2 --voice_over
```
If you run ```python3 inference_device.py -h``` you will see the helping guide which will aid you to understand what each argument means (as shown below).
```
usage: inference_device.py [-h] --device DEVICE --model_name MODEL_NAME
                           --confidence_thresh CONFIDENCE_THRESH
                           [--voice_over]
                           image_file

Infers on images and can give optional voice_over

positional arguments:
  image_file            A file path to the image

optional arguments:
  -h, --help            show this help message and exit
  --device DEVICE       name of device used
  --model_name MODEL_NAME
                        name of models: [efficientdet_d0,
                        mobilenet_fasterrcnn, ssdlite_mobilenet]
  --confidence_thresh CONFIDENCE_THRESH
                        value for confidence thresholding in nms
  --voice_over          choice to include user interface. Default is False
```

## Running the Webcam Demo (Easy Implementation)

After git cloning the repo you will have access to the file **Try_This_Detection_Demo.ipynb**\
To run this, just follow the simple steps outlined below (Note: You won't need any coding experience to run this):
 * Type Google Colab in your search engine
 * Once you open the first link, you will see a box that allows you the chance to create a new notebook, upload, along with other choices
 * Click on the tab that says Upload and upload your file that you git cloned into your local computer
 * Then, you will see two cells with a play button on each.
 * You will press both play buttons one after another in order. When you reach the last play button **you will see a webcam application that utilizes our software**.
