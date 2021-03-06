import numpy as np
import pandas as pd


def kursawe(num_samples: int = 100, dist: str = "Random") -> list:
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
    x = np.random.rand(num_samples, 3)
    for i in range(2):
        f1 = f1 - 10 * np.exp(-0.2 * np.sqrt(x[i] * x[i] + x[i + 1] * x[i + 1]))
    for i in range(3):
        f2 = f2 + np.power(np.abs(x[i]), 0.8) + 5 * np.power(np.sin(x[i]), 3)
    return (x, [f1, f2])


def four_bar_plane_truss(num_samples: int = 100) -> list:
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
    x = np.random.rand(num_samples, 4)
    F = 10
    E = 200000
    L = 200
    sigma = 10
    x[:, 0] = x[:, 0] * 2 * F / sigma + F / sigma
    x[:, 1] = (x[:, 1] * (3 - np.sqrt(2)) + np.sqrt(2)) * F / sigma
    x[:, 2] = (x[:, 2] * (3 - np.sqrt(2)) + np.sqrt(2)) * F / sigma
    x[:, 3] = x[:, 3] * 2 * F / sigma + F / sigma
    f1 = L * (2 * x[0] + np.sqrt(2 * x[1]) + np.sqrt(x[2]) + x[3])
    f2 = F * (L / E) * 2 * (1 / x[0] + 1 / x[3] + np.sqrt(2) * (1 / x[1] - 1 / x[2]))
    return (x, [f1, f2])


def gear_train_design(num_samples: int = 100) -> float:
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
    x = np.rand.randint(12, 61, (num_samples, 4))
    return (x, [np.square((1 / 6.931) - (x[0] * x[1]) / (x[2] * x[3]))])


def pressure_vessel(num_samples: int = 100) -> float:
    """Pressure Vessel design problem.

    As found in An augmented lagrange multiplier....

    Parameters
    ----------
    x : list or ndarray
        should contain 4 elements. First two should be discrete multiples or 0.0625.
        Last two should be continuous.

    Returns
    -------
    float
        cost
    """
    x = np.hstack(
        (
            np.random.randint(1, 10, (num_samples, 2)) * 0.0625,
            np.random.random((num_samples, 2)) * 80 + 50,
        )
    )
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
    G = -np.pi() * x3 * x3 * x4 + (4 / 3) * np.pi() * x3 * x3 * x3 + 1296000
    return (x, [F, G])


def speed_reducer(num_samples: int = 100) -> list:
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
    x = np.random.random((num_samples, 7))
    x[:, 0] = x[:, 0] * (3.6 - 2.6) + 2.6
    x[:, 1] = x[:, 1] * (0.8 - 0.7) + 0.7
    x[:, 2] = x[:, 2] * (11) + 17
    x[:, 3] = x[:, 3] * (1) + 7.3
    x[:, 4] = x[:, 4] * (1) + 7.3
    x[:, 5] = x[:, 5] * (1) + 2.9
    x[:, 6] = x[:, 6] * (0.5) + 5
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    x6 = x[5]
    x7 = x[6]
    f1 = (
        0.7854 * x1 * x2 * x2 * (10 * x3 * x3 / 3 + 14.933 * x3 - 43.0934)
        - 1.508 * x1 * (x6 * x6 + x7 * x7)
        + 7.477 * (x6 * x6 * x6 + x7 * x7 * x7)
        + 0.7854 * (x4 * x6 * x6 + x5 * x7 * x7)
    )
    f2 = np.sqrt(np.power(745 * x4 / (x2 * x3), 2) + 1.69 * np.power(10, 7)) / (
        0.1 * x6 * x6 * x6
    )
    f3 = np.sqrt(np.power(745 * x5 / (x2 * x3), 2) + 1.575 * np.power(10, 8)) / (
        0.1 * x7 * x7 * x7
    )
    return (x, [f1, f2, f3])


def welded_beam_design(num_samples: int = 100):
    """The Welded beam design
    
    AS found in An improved harmony search algo...
    
    Parameters
    ----------
    num_samples : int, optional
        Number of samples (the default is 100, which generates 100 datapoints)
    
    """
    x = np.random.rand(num_samples, 4)
    x[:, 0] = x[:, 0] * (0.125)
    x[:, 1] = x[:, 1] * (9.9) + 0.1
    x[:, 2] = x[:, 2] * (9.9) + 0.1
    x[:, 3] = x[:, 3] * (4.9) + 0.1
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    f = 1.10471 * x1 * x1 * x2 + 0.04811 * x3 * x4 * (14 + x2)
    return (x, [f])


def unconstrained_f(num_samples: int = 100):
    x = np.random.rand(num_samples, 2) * 100 - 50
    x1 = x[0]
    x2 = x[1]
    f1 = (
        np.exp(0.5 * np.square(x1 * x1 + x2 * x2 - 25))
        + np.power(np.sin(4 * x1 - 3 * x2), 4)
        + 0.5 * (2 * x1 + x2 - 10)
    )
    f2 = (
        1
        + np.square(x1 + x2 + 1)
        * (19 - 14 * x1 + 3 * x1 * x1 - 14 * x2 + 6 * x1 * x2 + 3 * x2 * x2)
    ) * (
        30
        + np.square(2 * x1 - 3 * x2)
        * (18 - 32 * x1 + 12 * x1 * x1 + 48 * x2 - 36 * x1 * x2 + 27 * x2 * x2)
    )
    return (x, [f1, f2])


def main():
    # decision_vars = np.random.randint(12, 61, (100, 4))
    """decision_vars = np.hstack(
        (np.random.randint(1, 10, (100, 2))*0.0625, np.random.random((100, 2))*80+50)
    )"""

    objectives = np.asarray(
        list(speed_reducer(individual) for individual in decision_vars)
    )
    data1 = np.hstack((decision_vars, np.reshape(objectives[:, 0], (100, 1))))
    data2 = np.hstack((decision_vars, np.reshape(objectives[:, 1], (100, 1))))
    data1 = pd.DataFrame(data1)
    data2 = pd.DataFrame(data2)
    data1.to_csv("speed_reducer_1.csv", index=False)
    data2.to_csv("speed_reducer_2.csv", index=False)


if __name__ == "__main__":
    main()

""" # Fourbar
    F = 10
    E = 200000
    L = 200
    sigma = 10
    c = F/sigma
    decision_vars[:, 0] =  decision_vars[:, 0] * (3*c- c) + c
    decision_vars[:, 1] =  decision_vars[:, 1] * (3*c- np.sqrt(2)*c) + np.sqrt(2)*c
    decision_vars[:, 2] =  decision_vars[:, 2] * (3*c- np.sqrt(2)*c) + np.sqrt(2)*c
    decision_vars[:, 3] =  decision_vars[:, 3] * (3*c- c) + c
"""
