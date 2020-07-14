
# MicaSense Batch Processing
This package provides a command line utility that allows for the batch processing of MicaSense Altum/RedEdge imagery into stacked TIFFs as outlined in the [Batch Processing notebook](https://micasense.github.io/imageprocessing/Batch%20Processing.html).

## Table of Contents
1. [Setup](#setup)

&nbsp; &nbsp; &nbsp; &nbsp; 1.1 [Installation](#installation)

&nbsp; &nbsp; &nbsp; &nbsp; 1.2 [Optional Configuration](#optional-configuration)

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 1.2.1 [Unix-like Systems (GNU/Linux, Mac OS X, etc.)](#unix-config)

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 1.2.2 [Windows Systems](#windows-config)

3. [Usage](#usage)

&nbsp; &nbsp; &nbsp; &nbsp; 3.1 [Required Argument](#required-argument)

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 3.1.1 [```imagepath```](#imagepath)

&nbsp; &nbsp; &nbsp; &nbsp; 3.2 [Optional Arguments](#optional-arguments)

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 3.2.1 [```-p```/```--panelpath```](#panelpath)

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 3.2.2 [```-o```/```--outputpath```](#outputpath)

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 3.2.3 [```-r```/```--referenceimage```](#referenceimage)

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 3.2.4 [```-pl```/```--pyramidlevels```](#pyramidlevels)

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 3.2.5 [```-t```/```--thumbnails```](#thumbnails)

## <span id="setup">Setup</span>
### <span id="installation">Installation</span>
1. Follow the [imageprocessing setup tutorial](https://micasense.github.io/imageprocessing/MicaSense%20Image%20Processing%20Setup.html) to ensure that all dependencies are installed within your system.

2. Activate the micasense Anaconda environment.
```bash
conda activate micasense
```
3. Run the install process for the ```micasense``` package from the **top level directory** of this repository.
```bash
cd imageprocessing
python setup.py install
```
You will now have both the ```micasense``` and ```batchprocessing``` packages installed within the micasense Anaconda environment.

You can check whether the install has been successful by running
```bash
python -m batchprocessing -h
```
from within the micasense Anaconda environment. This command should show the help printout for the ```batchprocessing``` utility if installed correctly.

### <span id="optional-configuration">Optional Configuration</span>
Following installation, the ```batchprocessing``` utility can be executed by first activating the micasense Anaconda environment, and then running the ```batchprocessing``` module installed within the environment - (see [Usage](#usage) for further detail).
```bash
conda activate micasense
python -m batchprocessing imagepath [options]
```
This level of verbosity can be a pain point for some users who wish to process imagery in a more streamlined manner - i.e. by simply running

```bash
batchprocessing imagepath [options]
```
#### <span id="unix-config">Unix-like Systems (GNU/Linux, Mac OS X, etc.)</span>
The above described workflow can be achieved by adding a function to the ```.bashrc``` (GNU/Linux) or ```.zshrc``` (Mac OS X 10.3+) file within your system.

```.bashrc```/```.zshrc``` files are stored in the user home directory by default. Check the relevant file for your operating system exists.
```bash
cd ~
ls -a # -a flag is important to list hidden files (of which target config files are)
```
If the relevant file does not exists then create it.

Append the following code to your ```.bashrc```/```.zshrc``` file.
```bash
function batchprocessing {
	conda activate micasense;
	python -m batchprocessing $@;
	conda deactivate;
}
```
Now open a new terminal session and run.
```bash
batchprocessing -h
```
You should see the help output of the ```batchprocessing``` utility print out to the console.

#### <span id="windows-config">Windows Systems</span>
The same functionality can be made available on Windows systems by carrying out the following changes to your PowerShell profile.

Launch Anaconda PowerShell Prompt and run the following command.
```powershell
Test-Path $profile
```
If the console printout following this command is ```True```, continue to ***Editing PowerShell Profile***.

If the console printout reads ```False```, run the following command (accepting the prompt to create a new PowerShell profile if one is raised).
```powershell
New-Item -path $profile -type file –force
```

##### Editing PowerShell Profile
Run the following command to print the location of the file which defines your PowerShell profile.
```powershell
$profile
```
Open this file in a text-editor and append the following function to it.
```powershell
Function batchprocessing
{
	conda activate micasense;
	python -m batchprocessing $args;
}
```
Now open a new terminal session and run.
```powershell
batchprocessing -h
```
You should see the help output of the ```batchprocessing``` utility print out to the console.

## <span id="usage">Usage</span>
This usage guide will make reference to the default command used to run ```batchprocessing``` following basic installation throughout for full clarity. i.e.
```bash
conda activate micasense
python -m batchprocessing imagepath [options]
```
If you have completed the [Optional Configuration](#optional-configuration) as outlined above, this command can be substituted with the less verbose
```bash
batchprocessing imagepath [options]
```
All examples used within this document make use of the path to a directory with the following structure as the ```imagepath``` argument.
```bash
ALTUM1SET/
├── 000
│   ├── IMG_0000_1.tif
│   ├── IMG_0000_2.tif
│   ├── IMG_0000_3.tif
│   ├── IMG_0000_4.tif
│   ├── IMG_0000_5.tif
│   ├── IMG_0000_6.tif
│   ├── IMG_0008_1.tif
│   ├── IMG_0008_2.tif
│   ├── IMG_0008_3.tif
│   ├── IMG_0008_4.tif
│   ├── IMG_0008_5.tif
│   └── IMG_0008_6.tif
└── 001
    ├── IMG_0245_1.tif
    ├── IMG_0245_2.tif
    ├── IMG_0245_3.tif
    ├── IMG_0245_4.tif
    ├── IMG_0245_5.tif
    └── IMG_0245_6.tif
```
Configuration for batch processing of imagery will be printed to the console at runtime. If upon this printing you realise that you are unhappy with elements of the configuration you can press ```ctrl + z``` to stop the process in progress, make changes to the configuration by changing the arguments you pass to the utility and run it again.

Example of runtime configuration printout.

```bash
BatchProcess Configuration
start time: 2020-07-10 12:35:57.527705
image path: /home/user/Desktop/ALTUM1SET
output path: /home/user/Desktop/ALTUM1SET/stacks
thumbnails: True
thumbnail path: /home/user/Desktop/ALTUM1SET/thumbnails
pyramidlevels: 0
panel group:
	/home/user/Desktop/ALTUM1SET/000/IMG_0000_3.tif
	/home/user/Desktop/ALTUM1SET/000/IMG_0000_4.tif
	/home/user/Desktop/ALTUM1SET/000/IMG_0000_2.tif
	/home/user/Desktop/ALTUM1SET/000/IMG_0000_6.tif
	/home/user/Desktop/ALTUM1SET/000/IMG_0000_1.tif
	/home/user/Desktop/ALTUM1SET/000/IMG_0000_5.tif
reference group:
	/home/user/Desktop/ALTUM1SET/000/IMG_0008_3.tif
	/home/user/Desktop/ALTUM1SET/000/IMG_0008_4.tif
	/home/user/Desktop/ALTUM1SET/000/IMG_0008_1.tif
	/home/user/Desktop/ALTUM1SET/000/IMG_0008_5.tif
	/home/user/Desktop/ALTUM1SET/000/IMG_0008_2.tif
	/home/user/Desktop/ALTUM1SET/000/IMG_0008_6.tif
```

The below examples are highly specific in that each only addresses the argument being discussed, this approach has been chosen as to minimise confusion and maximise readability when referencing this document. As many (or indeed as few) optional arguments may be passed at run time as in the following example.
```bash
conda activate micasense
python -m batchprocessing /home/user/Desktop/ALTUMSET1 -p /home/user/Desktop/ALTUMSET1/000/IMG_0000_1.tif -o /home/user/Desktop/10jun_out -r /home/user/Desktop/ALTUMSET1/001/IMG_0245_1.tif -pl 1 -t
```
### <span id="required-argument">Required Argument</span>
The below outlined argument (```imagepath```) is required for the utility to run to successful completion. The argument is positional and must be passed as the first argument at runtime when this utility is envoked.
#### <span id="imagepath">```imagepath```</span>
The ```imagepath``` argument passed at is used to specify the top-level directory of MicaSense Altum/RedEdge imagery to be processed into stacked TIFFs. This argument is positional and is defined as being the first argument passed to the ```batchprocessing``` utility.
```bash
conda activate micasense
python -m batchprocessing /home/user/Desktop/ALTUMSET1
```

The ```batchprocessing``` utility will process all imagery present in the above example directory, including all sub-directories contained within it.

Following execution the ```imagepath``` directory looks as follows.

```bash
ALTUM1SET/
├── 000
│   ├── IMG_0000_1.tif
│   ├── IMG_0000_2.tif
│   ├── IMG_0000_3.tif
│   ├── IMG_0000_4.tif
│   ├── IMG_0000_5.tif
│   ├── IMG_0000_6.tif
│   ├── IMG_0008_1.tif
│   ├── IMG_0008_2.tif
│   ├── IMG_0008_3.tif
│   ├── IMG_0008_4.tif
│   ├── IMG_0008_5.tif
│   └── IMG_0008_6.tif
├── 001
│   ├── IMG_0245_1.tif
│   ├── IMG_0245_2.tif
│   ├── IMG_0245_3.tif
│   ├── IMG_0245_4.tif
│   ├── IMG_0245_5.tif
│   └── IMG_0245_6.tif
├── stacks
│   ├── 8Zo8hoxho012MQT5vm02.tif
│   ├── imageSet.json
│   ├── JD5rHuGQxlLsp4EJRBaJ.tif
│   ├── log.csv
│   └── mWuludZ3pUdfNBQddCQd.tif
└── thumbnails
    ├── 8Zo8hoxho012MQT5vm02.jpg
    ├── JD5rHuGQxlLsp4EJRBaJ.jpg
    └── mWuludZ3pUdfNBQddCQd.jpg
```
Processed/stacked TIFFs are saved to the ```stacks``` directory alongside the ```imageSet.json``` and ```log.csv``` files containing metadata written to the stacked image files.

Previews of processed images are written to the ```thumbnails``` directory in JPEG format for reference. See [```-t/--thumbnails```](#thumbnails) for guidance on bypassing this feature.

### Optional Arguments
The below outlined arguments are entirley optional, allowing for greater control over parameters used when processing imagery using this tool as desired.

#### <span id="panelpath">```-p```/```--panelpath```</span>
The ```--panelpath``` argument defines a panel capture to be used when processing MicaSense Altum/RedEdge imagery. The argument passed at runtime should be the system path to a single image from the panel capture group to be used e.g.
```bash
conda activate micasense
python -m batchprocessing /home/user/Desktop/ALTUMSET1 -p /home/user/Desktop/ALTUMSET1/000/IMG_0000_1.tif
```
If this argument is not present at runtime, the ```batchprocessing``` utility will search the directory defined by the ```imagepath``` argument and it's sub-directories recursivley to identify a panel capture group to be used whilst processing imagery. A panel capture group is defined by it's naming using the IMG_0000_*.tif convention.

#### <span id="outputpath">```-o```/```--outputpath```</span>
The ```--outputpath``` argument specifies the path to a location in which processed TIFFs and preview images (if applicable, see ```-t/--thumbnails```) will be written. In the below example the path passed as this argument is an empty directory.
```bash
conda activate micasense
python -m batchprocessing /home/user/Desktop/ALTUMSET1 -o /home/user/Desktop/10jun_out
```
Following execution, the target directory looks as follows.
```bash
10jun_out/
├── stacks
│   ├── 8Zo8hoxho012MQT5vm02.tif
│   ├── imageSet.json
│   ├── JD5rHuGQxlLsp4EJRBaJ.tif
│   ├── log.csv
│   └── mWuludZ3pUdfNBQddCQd.tif
└── thumbnails
    ├── 8Zo8hoxho012MQT5vm02.jpg
    ├── JD5rHuGQxlLsp4EJRBaJ.jpg
    └── mWuludZ3pUdfNBQddCQd.jpg
```
If this argument is not passed at runtime the ```stacks``` and ```thumbnails``` directories and containing files will be written to the directory defined by the ```imagepath``` argument.

#### <span id="referenceimage">```-r```/```--referenceimage```</span>
The ```--referenceimage``` argument constitutes the path to an image file from the capture group to be used in generating parameters for usage in processing imagery at runtime.

Guidance on selecting appropriate images to be passed as this argument can be found [here](https://micasense.github.io/imageprocessing/Alignment.html).

Example usage.
```bash
conda activate micasense
python -m batchprocessing /home/user/Desktop/ALTUMSET1 -r /home/user/Desktop/ALTUMSET1/001/IMG_0245_1.tif
```

#### <span id="pyramidlevels">```-pl```/```--pyramidlevels```</span>
The ```--pyramidlevels``` argument specifies how many "levels" the code will execute when processing imagery using a homographic approach (see [this article](https://www.learnopencv.com/homography-examples-using-opencv-python-c/) for examples of homography using the Python ```opencv``` library). Experimenting with this parameter can sometimes help to more precisely align images that are taken of subjects in close proximity. [This post](https://stackoverflow.com/questions/45997891/cv2-motion-euclidean-for-the-warp-mode-in-ecc-image-alignment-method) on StackOverflow features discussion of a "pyramid levels" approach to homographic image processing similar to that employed in this tool and the wider MicaSense imageprocessing repository.

If this argument is not present at runtime the default value applied to the ```pyramidlevels``` variable is 0.

Example usage.
```bash
conda activate micasense
python -m batchprocessing /home/user/Desktop/ALTUMSET1 -pl 1
```
#### <span id="thumbnails">```-t```/```--thumbnails```
The ```--thumbnails``` argument is a simple flag used to specify that a directory named ```thumbnails``` containing JPEG previews of processed imagery will **not** be written at runtime. i.e. Thumbnails will be generated by default if this flag is not raised and will not be included if it is.

Example usage.
```bash
conda activate micasense
python -m batchprocessing /home/user/Desktop/ALTUMSET1 -t
```

Following execution, the directory specified by the ```imagepath``` argument will look as follows.
```bash
ALTUM1SET/
├── 000
│   ├── IMG_0000_1.tif
│   ├── IMG_0000_2.tif
│   ├── IMG_0000_3.tif
│   ├── IMG_0000_4.tif
│   ├── IMG_0000_5.tif
│   ├── IMG_0000_6.tif
│   ├── IMG_0008_1.tif
│   ├── IMG_0008_2.tif
│   ├── IMG_0008_3.tif
│   ├── IMG_0008_4.tif
│   ├── IMG_0008_5.tif
│   └── IMG_0008_6.tif
├── 001
│   ├── IMG_0245_1.tif
│   ├── IMG_0245_2.tif
│   ├── IMG_0245_3.tif
│   ├── IMG_0245_4.tif
│   ├── IMG_0245_5.tif
│   └── IMG_0245_6.tif
└── stacks
    ├── 8Zo8hoxho012MQT5vm02.tif
    ├── imageSet.json
    ├── JD5rHuGQxlLsp4EJRBaJ.tif
    ├── log.csv
    └── mWuludZ3pUdfNBQddCQd.tif
```
