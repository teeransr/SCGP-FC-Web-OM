from cx_Freeze import setup, Executable
import sys
import os
import scipy

os.environ['TCL_LIBRARY'] = "C:\\ProgramData\\Anaconda3\\tcl\\tcl8.6"
os.environ['TK_LIBRARY'] = "C:\\ProgramData\\Anaconda3\\tcl\\tk8.6"

base = None

if sys.platform == 'win32':
    base = 'Win32GUI'

scipy_path = os.path.dirname(scipy.__file__)

executables = [Executable("forecastModel.py", base=base)]

packages = ["idna", "numpy","pandas","tkinter","sklearn.model_selection","sklearn.metrics","statsmodels.tsa.arima_model","threading","warnings","time","datetime","dateutil.relativedelta"]
options = {
    'build_exe': {

        'packages':packages,
        'include_files': ["tcl86t.dll", "tk86t.dll", scipy_path]
    },

}

setup(
    name = "Main",
    options = options,
    version = "1.1",
    description = 'Nothing',
    executables = executables, requires=['numpy', 'pandas', 'dateutil', 'scikit-learn', 'statsmodels']
)