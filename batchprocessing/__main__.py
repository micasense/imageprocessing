MicaSenseimport argparse
import os
import sys
from copy import copy

SRC_ROOT = os.path.abspath(os.path.dirname(sys.argv[0]))
sys.path.insert(0, SRC_ROOT)

from batch_processing import BatchProcess


parser = argparse.ArgumentParser(prog='batchprocessing',
    usage='python -m %(prog)s imagepath [options]',
    description='Utility to align and stack MicaSense imagery.',
    epilog=None)

parser.add_argument('imagepath', type=str, metavar='imagepath',
    help='Full path to a top-level directory containing MicaSense image captures to be aligned and stacked.')

parser.add_argument('-p', '--panelpath', required=False, type=str, metavar='panelpath',
    help='''Path to an appropriate panel image used for the capture(s) specified by the imagepath argument. Any image from the capture group is fine
    e.g. PATH/IMG_0000_1.tif, PATH/IMG_0000_2.tif etc. as the CLI groups images to captures based on the IMG_XXXX
    part of the filename. If this argument is not specified, the program defaults to searching for a panel capture within the imagepath
    directory/sub-directories, as identified by the IMG_0000_*.tif nomenclature used to signify panel images. If not such capture is found during this
    search, the program continues with no panel capture/images to reference.''')

parser.add_argument('-o', '--outputpath', required=False, type=str, metavar='outputpath',
    help='''Full path to the location in which the aligned stack and thumbnail directories will be written.
    If this argument is not provided, the aligned "stacks" and "thumbnails" directories will be written to the imagepath provided at runtime.''')

parser.add_argument('-r', '--referenceimage', required=False, type=str, metavar='referenceimage',
    help='''Full path to capture image that will be used to generate warp matrices enabling stack alignment e.g. /home/user/documents/data/0000SET/000/IMG_0001_1.tif -
    Any image from the capture group is fine e.g. PATH/IMG_0001_1.tif, PATH/IMG_0001_2.tif etc. as the CLI groups images to captures based on the IMG_XXXX
    part of the filename. If this argument is not passed, a pseudorandom capture from the --imagepath directory/sub-directories will be used to generate
    the warp matrices to enable image/stack alignment. The pseudorandom capture will not be a panel capture as identified by the IMG_0000_*.tif panel capture naming convention.''')

parser.add_argument('-pl', '--pyramidlevels', required=False, type=int, metavar='pyramidlevels', default=0,
    help='''Value to specify how many "levels" the code will execute when using the pyramid approach to align images.
    This can sometimes help to more precisely align images that are taken of subjects in close proximity. The default value is 0. ''')

parser.add_argument('-t', '--thumbnails', required=False, action='store_false', default=True,
    help='''Raising this flag will run the program WITHOUT creating a "thumbnails" directory containing .jpg previews of output stacks
    within the output directory - if this flag is not present the thumbnails directory and files will be written by default.''')

args = parser.parse_args()


kwargs = args.__dict__
imagepath = kwargs['imagepath']
del kwargs['imagepath']
cli_process = BatchProcess(imagepath, **args.__dict__)

if cli_process.process_complete:
    print('Done')
else:
    print('\nError raised in batchprocessing of imagery, please see above traceback.')
