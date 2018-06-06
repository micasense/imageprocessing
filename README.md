# README 

### MicaSense RedEdge Image Processing Tutorials 

This repository includes tutorials and examples for processing MicaSense RedEdge images into usable information using the python programming language.  The intended audience is researchers and developers with some software development experience that want to do their own image processing.  While a number of commercial tools fully support processing RedEdge data into reflectance maps, there are a number of reasons to process your own data, including controlling the entire radiometric workflow (for academic or publication reasons), pre-processing images to be used in a non-radiometric photogrammetry suite, or processing single sets of 5 images without building a larger map.

### How do I get set up? 

First, `git clone` this repository, as it has all of the code and examples you'll need.

To do that you'll need [git](https://git-scm.com/downloads)

Once you have git installed and the repository cloned, you are ready to start with the first tutorial. Check out the [setup tutorial](https://micasense.github.io/imageprocessing/MicaSense%20Image%20Processing%20Setup.html) which will walk through installing and checking the necessary tools to run the remaining tutorials.

### Tutorial Articles

1. [MicaSense Image Processing Setup](https://micasense.github.io/imageprocessing/MicaSense%20Image%20Processing%20Setup.html)
1. [MicaSense Image Processing Tutorial #1](https://micasense.github.io/imageprocessing/MicaSense%20Image%20Processing%20Tutorial%201.html) (basic radiometic corrections)
1. [MicaSense Image Processing Tutorial #2](https://micasense.github.io/imageprocessing/MicaSense%20Image%20Processing%20Tutorial%202.html) (library introduction)
1. [MicaSense Image Processing Tutorial #3](https://micasense.github.io/imageprocessing/MicaSense%20Image%20Processing%20Tutorial%203.html) (basic DLS processing)
1. [Image Class Examples](https://micasense.github.io/imageprocessing/Images.html)
1. [Capture Class Examples](https://micasense.github.io/imageprocessing/Captures.html)
1. [ImageSet Examples](https://micasense.github.io/imageprocessing/ImageSets.html)
1. [Capture Alignment and Analysis Examples](https://micasense.github.io/imageprocessing/Alignment.html)


### MicaSense Library Usage

In addition to the tutorials, we've created library code that shows some common transformations, usages, and applications of RedEdge imagery.  In general, these are intended for developers that are familiar with installing and managing python packages and third party software.  The purpose of this code is readability and clarity to help others develop processing workflows, therefore performance may not be optimal.

While this code is similar to an installable python library (and we may support the `pip install` process in the future) the main purpose of this library is one of documentation and education. For this reason, we expect most users to be looking at the source code for understanding or improvement, and because of this you will currently need to run your notebooks from the directory you `git clone`d it into. 

### Running this code

The code in these tutorials consists of two parts. First, the tutorials generally end in `.ipynb` and are the Jupyter notebooks that were used to create the web page tutorials linked above. You can run this code by opening a terminal/iTerm (linux/mac) or Anaconda Command Prompt (Windows), navigating to the folder you cloned the git repository into, and running

```bash
jupyter notebook .
```

That command should open a web browser window showing the set of files and folder in the repository. Click the `...Setup.ipynb` notebook to get started.

Second, a set of helper utilities is available in the `micasense` folder that can be used both with these tutorials as well as separtely. 

### Contribution guidelines

Find a problem with the tutorial? Please create an issue on github. 

Want to correct an issue or expand library functionality?  Fork the repository, make your fix, and submit a pull request on github.

Have a question? Double-check that you're able to run the setup notebook successfully, and then check the [MicaSense Knowledgebase](https://support.micasense.com) before contacting support.

### Tests

Tests for many library functions are included in the `tests` diretory. Install the `pytest` module through your package manager (e.g. `pip install pytest`) and then tests can be run from the main directory using the command:

```bash
pytest tests
```

Data used by the tests is included in the `data` folder.

### For (Tutorial) Developers 

To generate the HTML pages after updating the jupyter notebooks, run the following command in the repository directory:

```bash
jupyter nbconvert --to html --ExecutePreprocessor.timeout=None --output-dir docs --execute *.ipynb
```

## License

The MIT License (MIT)

Copyright (c) 2017 MicaSense, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
