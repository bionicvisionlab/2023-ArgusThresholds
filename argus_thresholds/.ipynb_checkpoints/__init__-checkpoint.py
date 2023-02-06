# This import is necessary to ensure consistency of the generated images across
# platforms, and for the tests to run on Travis:
# https://stackoverflow.com/questions/35403127/testing-matplotlib-based-plots-in-travis-ci
# http://www.davidketcheson.info/2015/01/13/using_matplotlib_image_comparison.html
import matplotlib
matplotlib.use('agg')

from .info import __version__  # noqa
from .core import *  # noqa

from . import model
from . import cv
from . import metrics
from . import utils
from . import viz
