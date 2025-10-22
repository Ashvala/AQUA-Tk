import numpy as np
from .utils import BARK as bark
from .utils import module, HANN
from dataclasses import dataclass
from numba import njit

T100 = 0.03
Tmin = 0.008

@dataclass
class ModulationIn:
    """
    Initializes an instance of the ModulationIn class.

    Args:
        e2_tmp (float): The value of the e2_tmp parameter.
        etilde_tmp (float): The value of the etilde_tmp parameter.
        eder_tmp (float): The value of the eder_tmp parameter.
    """
    def __init__(self, e2_tmp, etilde_tmp, eder_tmp):
        """
        Initializes an instance of the Class.

        Args:
            e2_tmp: The value of e2_tmp parameter.
            etilde_tmp: The value of etilde_tmp parameter.
            eder_tmp: The value of eder_tmp parameter.
        """
        self.E2tmp = e2_tmp
        self.Etildetmp = etilde_tmp
        self.Edertmp = eder_tmp

    def __repr__(self):
        """
        Returns a string representation of the object.

        Returns:
        str: A string representing the object in the format:
             "ModulationIn(E2tmp={E2tmp}, Etildetmp={Etildetmp}, Edertmp={Edertmp})"
        """
        return f"ModulationIn(E2tmp={self.E2tmp}, Etildetmp={self.Etildetmp}, Edertmp={self.Edertmp})"


def modulation(E2, rate, in_struct, fC):
    """
    Args:
        E2: A list or array which contains the values of E2.
        rate: The modulation rate.
        in_struct: An object of a structure/class that contains the following attributes:
            - Edertmp: A list or array for storing temporary values of Edertmp.
            - E2tmp: A list or array for storing temporary values of E2tmp.
            - Etildetmp: A list or array for storing temporary values of Etildetmp.
        fC: A list or array which contains the values of fC.

    Returns:
        A tuple containing two elements:
        - Mod: A list or array which contains the calculated modulation values.
        - in_struct: The updated in_struct object with updated values of Edertmp, E2tmp, and Etildetmp.
    """
    Mod = np.zeros(bark)
    for k in range(bark):
        T = Tmin + (100.0 / fC[k]) * (T100 - Tmin)
        a = np.exp(-HANN / (2 * rate * T))
        in_struct.Edertmp[k] = (
            in_struct.Edertmp[k] * a
            + (1 - a) * (rate / (HANN / 2)) * module(E2[k]**0.3 - in_struct.E2tmp[k]**0.3)
        )
        in_struct.E2tmp[k] = E2[k]
        in_struct.Etildetmp[k] = a * in_struct.Etildetmp[k] + (1 - a) * E2[k]**0.3

        Mod[k] = in_struct.Edertmp[k] / (1 + (in_struct.Etildetmp[k] / 0.3))

    return Mod, in_struct
