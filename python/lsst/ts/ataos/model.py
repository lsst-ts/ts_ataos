
import numpy as np
from numpy.polynomial.polynomial import polyval

__all__ = ['Model']


class Model:
    """ A model class to that handles the ATAOS application level code. This
    class implements the corrections for each of the axis on the AT telescope,
    e.g. M1 and M2 pressure and hexapod correction.

    The AOS corrections are simple polynomes where x is the gravity vector,
    cos(theta), where theta is the azimuth distance or 90.-altitude.

    """
    def __init__(self):
        self.config = {'m1': [0.0],
                       'm2': [0.0],
                       'hexapod_x': [0.0],
                       'hexapod_y': [0.0],
                       'hexapod_z': [0.0],
                       'hexapod_u': [0.0],
                       'hexapod_v': [0.0]
                       }

    def get_correction_m1(self, azimuth, elevation, temperature=None):
        """Correction for m1 support pressure.

        Parameters
        ----------
        azimuth : `float`
            Azimuth position for the correction (degrees).
        elevation : `float`
            Elevation position for correction (degrees). Currently ignored.
        temperature : `float`
            Temperature for correction (C). Currently ignored.

        Returns
        -------
        pressure : float
            Pressure to apply (Pascal).
        """
        return polyval(np.cos(np.radians(90. - elevation)),
                       self.config['m1'])

    def get_correction_m2(self, azimuth, elevation, temperature=None):
        """Correction for m2 support pressure.

        Parameters
        ----------
        azimuth : `float`
            Azimuth position for the correction (degrees).
        elevation : `float`
            Elevation position for correction (degrees). Currently ignored.
        temperature : `float`
            Temperature for correction (C). Currently ignored.

        Returns
        -------
        pressure : float
            Pressure to apply (Pascal).
        """
        return polyval(np.cos(np.radians(90. - elevation)),
                       self.config['m2'])

    def get_correction_hexapod(self, azimuth, elevation, temperature=None):
        """Correction for hexapod position.

        Parameters
        ----------
        azimuth : `float`
            Azimuth position for the correction (degrees).
        elevation : `float`
            Elevation position for correction (degrees). Currently ignored.
        temperature : `float`
            Temperature for correction (C). Currently ignored.

        Returns
        -------
        x : float
            x-axis position (um)
        y : float
            y-axis position (um)
        z : float
            z-axis position (um)
        u : float
            rotation angle with respect to x-axis (degrees)
        v : float
            rotation angle with respect to y-axis (degrees)
        w : float
            rotation angle with respect to z-axis (degrees)
        """
        x = polyval(np.cos(np.radians(90. - elevation)),
                    self.config['hexapod_x'])
        y = polyval(np.cos(np.radians(90. - elevation)),
                    self.config['hexapod_y'])
        z = polyval(np.cos(np.radians(90. - elevation)),
                    self.config['hexapod_z'])
        u = polyval(np.cos(np.radians(90. - elevation)),
                    self.config['hexapod_u'])
        v = polyval(np.cos(np.radians(90. - elevation)),
                    self.config['hexapod_v'])
        w = 0.

        return x, y, z, u, v, w
