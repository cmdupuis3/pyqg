__version__='0.1.3'
from .model import Model
from .qg_model import QGModel
from .bt_model import BTModel
from .sqg_model import SQGModel
from .layered_model import LayeredModel
from .particles import LagrangianParticleArray2D, GriddedLagrangianParticleArray2D
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pyqg")
except PackageNotFoundError:
    # package is not installed
    pass
