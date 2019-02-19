import numpy as np


def kursawe(x) -> list:
    """Kursawe Test function.

    As found in Multiobjective structural optimization using a microgenetic algorithm.

    Parameters
    ----------
    x : list or ndarray
        x is a vector with 3 components. -5 < Xi < 5

    Returns
    -------
    list
        Returns a list of f1 and f2.
    """
    f1 = 0
    f2 = 0
    for i in range(2):
        f1 = f1 - 10 * np.exp(-0.2 * np.sqrt(x[i] * x[i] + x[i + 1] * x[i + 1]))
    for i in range(3):
        f2 = f2 + np.power(np.abs(x[i]), 0.8) + 5 * np.power(np.sin(x[i]), 3)
    return (f1, f2)


def four_bar_plane_truss(x) -> list:
    """Four bar plane truss problem.

    As found in Multiobjective structural optimization using a microgenetic algorithm.

    Parameters
    ----------
    x : list or ndarray
        Should have 4 elements

    Returns
    -------
    list
        (f1, f2)
    """
    F = 10
    E = 200000
    L = 200
    sigma = 10
    f1 = L * (2 * x[0] + np.sqrt(2 * x[1]) + np.sqrt(x[2]) + x[3])
    f2 = F * (L / E) * 2 * (1 / x[0] + 1 / x[3] + np.sqrt(2) * (1 / x[1] - 1 / x[2]))
    return (f1, f2)


def gear_train_design(x) -> float:
    """Gear Train Design
    
    As found in Augmented Lagrange multiplier...
    
    Parameters
    ----------
    x : list or ndarray
        Should have 4 elements, integers in the range [12, 60].
    
    Returns
    -------
    float
    """
    return np.square(1 / 6.931 + (x[0] * x[1]) / (x[2] * x[3]))


def pressure_vessel(x) -> float:
    """Pressure Vessel design problem.

    As found in An augmented lagrange multiplier....

    Parameters
    ----------
    x : list or ndarray
        should contain 4 elements. First two should be discrete multiples or 0.625.
        Last two should be continuous.

    Returns
    -------
    float
        cost
    """
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    F = (
        0.6224 * x1 * x3 * x4
        + 1.7781 * x2 * x3 * x3
        + 3.1661 * x1 * x1 * x4
        + 19.84 * x1 * x1 * x3
    )
    return F


def speed_reducer(x) -> list:
    """Speed reducer problem. Biobjective.
    
    As found in Multiobjective structural optimization using a microgenetic algorithm.
    
    Parameters
    ----------
    x : list or ndarray
        7 element vector.
    
    Returns
    -------
    list
        weight and stress
    """
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    x6 = x[5]
    x7 = x[6]
    Fweight = (
        0.7854 * x1 * x2 * x2 * (10 * x3 * x3 / 3 + 14.933 * x3 - 43.0934)
        - 1.508 * x1 * (x6 * x6 + x7 * x7)
        + 7.477 * (x6 * x6 * x6 + x7 * x7 * x7)
        + 0.7854(x4 * x6 * x6 + x5 * x7 * x7)
    )
    Fstress = np.sqrt(np.pow(745 * x4 / (x2 * x3), 2) + np.pow(1.6910, 7)) / (
        0.1 * x6 * x6 * x6
    )
