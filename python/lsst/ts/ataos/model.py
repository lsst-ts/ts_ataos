
import numpy as np

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

        self.poly_m1 = np.poly1d(self.config['m1'])
        self.poly_m2 = np.poly1d(self.config['m2'])
        self.poly_x = np.poly1d(self.config['hexapod_x'])
        self.poly_y = np.poly1d(self.config['hexapod_y'])
        self.poly_z = np.poly1d(self.config['hexapod_z'])
        self.poly_u = np.poly1d(self.config['hexapod_u'])
        self.poly_v = np.poly1d(self.config['hexapod_v'])

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
        return self.poly_m1(np.cos(np.radians(90. - elevation)))

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
        return self.poly_m2(np.cos(np.radians(90. - elevation)))

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
        x = self.poly_x(np.cos(np.radians(90. - elevation)))
        y = self.poly_y(np.cos(np.radians(90. - elevation)))
        z = self.poly_z(np.cos(np.radians(90. - elevation)))
        u = self.poly_u(np.cos(np.radians(90. - elevation)))
        v = self.poly_v(np.cos(np.radians(90. - elevation)))
        w = 0.

        return x, y, z, u, v, w

    @property
    def m1(self):
        return self.config['m1']

    @m1.setter
    def m1(self, val):
        self.config['m1'] = val
        self.poly_m1 = np.poly1d(val)

    @property
    def m2(self):
        return self.config['m2']

    @m2.setter
    def m2(self, val):
        self.config['m2'] = val
        self.poly_m2 = np.poly1d(val)

    @property
    def hexapod_x(self):
        return self.config['hexapod_x']

    @hexapod_x.setter
    def hexapod_x(self, val):
        self.config['hexapod_x'] = val
        self.poly_x = np.poly1d(val)

    @property
    def hexapod_y(self):
        return self.config['hexapod_y']

    @hexapod_y.setter
    def hexapod_y(self, val):
        self.config['hexapod_y'] = val
        self.poly_y = np.poly1d(val)

    @property
    def hexapod_z(self):
        return self.config['hexapod_z']

    @hexapod_z.setter
    def hexapod_z(self, val):
        self.config['hexapod_z'] = val
        self.poly_z = np.poly1d(val)

    @property
    def hexapod_u(self):
        return self.config['hexapod_u']

    @hexapod_u.setter
    def hexapod_u(self, val):
        self.config['hexapod_u'] = val
        self.poly_u = np.poly1d(val)

    @property
    def hexapod_v(self):
        return self.config['hexapod_v']

    @hexapod_v.setter
    def hexapod_v(self, val):
        self.config['hexapod_v'] = val
        self.poly_v = np.poly1d(val)
