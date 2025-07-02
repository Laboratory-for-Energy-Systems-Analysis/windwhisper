from pathlib import Path

__version__ = (0, 0, 1)
__all__ = (
    "__version__",
    "DATA_DIR",
    "WindTurbines",
    "WindSpeed",
    "NoisePropagation",
    "NoiseAnalysis",
    "HumanHealth"
)

HOME_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(__file__).resolve().parent / "data"


from .windturbines import WindTurbines
from .windspeed import WindSpeed
from .noisepropagation import NoisePropagation
from .noiseanalysis import NoiseAnalysis
from .health_impacts import HumanHealth

