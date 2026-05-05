from dataclasses import dataclass
from typing import Tuple, Any
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

type number = float | int | np.number[Any]

@dataclass
class GraphVisuals:
    styles: list[str]
    axis_font_size: int = 0
    legend_font_size: int = 0
    N: int = 1

    def __getitem__(self, i) -> str:
        return self.styles[i]
    
    def __setitem__(self, i, style) -> None:
        self.styles[i] = style

@dataclass
class WeightedMeasurements:
    x: np.ndarray
    y: np.ndarray
    x_err_array: np.ndarray
    y_err_array: np.ndarray
    
    def __post_init__(self):
        if (len(self.x) != len(self.y)): raise ValueError("x and y arrays must have the same length.")

    @property
    def N(self):
        return len(self.x)
    
    def __getitem__(self, i: int) -> Tuple[Any, Any, Any, Any]:
        return self.x[i], self.y[i], self.x_err_array[i], self.y_err_array[i]
    
    def __setitem__(self, i: int, xy) -> None:
        self.x[i] = xy[0]
        self.y[i] = xy[1]
        self.x_err_array[i] = xy[2]
        self.y_err_array[i] = xy[3]

@dataclass
class Measurements:
    x: npt.NDArray[np.number]
    y: npt.NDArray[np.number]
    sigma_y: number = 0.

    def __post_init__(self):
        if (len(self.x) != len(self.y)): raise ValueError("x and y arrays must have the same length.")

    @property
    def N(self):
        return len(self.x)

    def __getitem__(self, i: int) -> Tuple[Any, Any]:
        return self.x[i], self.y[i]
    
    def __setitem__(self, i: int, xy) -> None:
        self.x[i] = xy[0]
        self.y[i] = xy[1]

@dataclass
class xyData:
    """
    Object storing arrays of point coordinates
    """
    x: npt.NDArray[Any]
    y: npt.NDArray[Any]

    def __getitem__(self, i: int) -> Tuple[Any, Any]:
        return self.x[i], self.y[i]
    
    def __setitem__(self, i: int, xy) -> None:
        self.x[i] = xy[0]
        self.y[i] = xy[1]

@dataclass
class RegData:
    """
    Class to store parameters about the results of a linear regression
    """
    a:     number = 0.
    b:     number = 0.
    a_err: number = 0.
    b_err: number = 0.
    chi2:  number = 0.
    nu:    number = 0.

def lin_reg(data: Measurements) -> Tuple[RegData, xyData]:
    """
    Function to calculate linear regression parameters

    Parameters
    ----------
    data : Measurements
        Data to fit

    Returns
    -------
    regression_data : RegData
        Object with the regression parameters

    xy_data : xyData
        Object that contains x and y of the points on the regression line
    """
    x = data.x
    y = data.y
    sigma_y = data.sigma_y

    N = data.N
    if (N != len(y)): raise Exception("ARRAYS HAVE NON-MATCHING SIZE (in lin_reg())")

    # Varie somme
    S_X = np.sum(x)
    S_XX = np.sum(x ** 2)
    S_Y = np.sum(y)
    S_XY = np.sum(x * y)

    # Delta
    D = N * S_XX - S_X ** 2

    # Pendenza del fit, iniziale
    a = 1 / D * (S_XX * S_Y - S_X * S_XY)
    b = 1 / D * (N * S_XY - S_X * S_Y)

    a_err = sigma_y * np.sqrt(S_XX / D)
    b_err = sigma_y * np.sqrt(N / D) # errore aggiornato

    # Chi quadro e gradi di libertà
    chi2 = np.sum(((y - (a + b * x)) / sigma_y)**2)
    nu = N - 2

    return RegData(a, b, a_err, b_err, chi2, nu), xyData(x, a + b * x)


def weightedLinReg(data: WeightedMeasurements, maxLoops: int = 0) -> Tuple[RegData, xyData]:
    """
    Function to calculate weighted linear regression parameters

    Parameters
    ----------
    data : WeightedMeasurements
        Data to fit

    maxLoops : int, optional
        Maximum iterations to finetune the slope

    Returns
    -------
    regression_data : RegData
        Object with the regression parameters

    xy_data : xyData
        Object that contains x and y of the points on the regression line
    """
    x = data.x
    y = data.y
    x_err = data.x_err_array
    y_err = data.y_err_array

    N = data.N
    if (len(y) != N or len(x_err) != N or len(y_err) != N): raise Exception("ARRAYS HAVE NON-MATCHING SIZE (in weightedLinReg())")

    # Varie somme
    W = 1 / (y_err ** 2) # "Pesi" - Weights
    S_W   = np.sum(W)
    S_XW  = np.sum(x * W)
    S_YW  = np.sum(y * W)
    S_XXW = np.sum((x ** 2) * W)
    S_XYW = np.sum(x * y * W)

    # Delta
    D_W = S_W * S_XXW - S_XW ** 2

    # Fit iniziale
    b = (1 / D_W) * (S_W * S_XYW - S_XW * S_YW)

    # migliorare
    y_err_i = np.sqrt(y_err ** 2 + (b * x_err) ** 2) # errore 2
    b_err = np.sqrt(S_W / D_W)
    b1 = b + 2 * b_err

    loop_count = 0
    while (abs(b - b1) > b_err) and (maxLoops==0 or loop_count < maxLoops): # loop di miglioramento
        y_err_i = np.sqrt(y_err ** 2 + (b * x_err) ** 2)
        # Varie somme
        W = 1 / (y_err_i ** 2) # "Pesi" - Weights
        S_W   = np.sum(W)
        S_XW  = np.sum(x * W)
        S_YW  = np.sum(y * W)
        S_XXW = np.sum((x ** 2) * W)
        S_XYW = np.sum(x * y * W)
        
        # Delta
        D_W = S_W * S_XXW - S_XW ** 2

        b_err = np.sqrt(S_W / D_W)
        b1 = b
        b = (1 / D_W) * (S_W * S_XYW - S_XW * S_YW)

        loop_count += 1
        if loop_count > 100: raise Exception("weightedLinReg reached 100 iterations: check your data, the code for this function, or use lin_reg()")

    # Intercetta
    a = (1 / D_W) * (S_XXW * S_YW - S_XW * S_XYW)

    # Errori sui parametri
    a_err = np.sqrt(S_XXW / D_W)
    b_err = np.sqrt(S_W / D_W) # errore finale
    y_err_i = np.sqrt(y_err ** 2 + (b * x_err) ** 2)

    # Chi quadro e gradi di libertà
    chi2 = np.sum(((y - (a + b * x)) / y_err_i) ** 2)
    nu = N - 2

    return RegData(a, b, a_err, b_err, chi2, nu), xyData(x, a + b * x)


def plotGraphs(data_array: list[WeightedMeasurements | Measurements | xyData], styles: GraphVisuals, maxLoops: int = 0) -> None:
    """
    Plot multiple graphs
    """
    for i, data in enumerate(data_array):
        style = styles[i]
        if type(data) == WeightedMeasurements:
            regression_data, xy_data = weightedLinReg(data, maxLoops)

