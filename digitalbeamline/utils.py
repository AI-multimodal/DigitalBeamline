import importlib

import matplotlib as mpl
import matplotlib.pyplot as plt

from digitalbeamline import HOME


def create_home_directory():
    HOME.mkdir(exist_ok=True, parents=True)


def get_function_from_signature(signature):
    """Parases a function of the form module.submodule:function to import
    and get the actual function as defined.

    Parameters
    ----------
    signature : str

    Returns
    -------
    callable
    """

    module, function = signature.split(":")
    module = importlib.import_module(module)
    return getattr(module, function)


def set_plotting_defaults():
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.family'] = 'STIXGeneral'
    mpl.rcParams['text.usetex'] = False
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('axes', labelsize=12)
    mpl.rcParams['figure.dpi'] = 300
