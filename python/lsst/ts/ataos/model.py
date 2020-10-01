import numpy as np
import logging

__all__ = ["Model"]


class Model:
    """ A model class to that handles the ATAOS application level code. This
    class implements the corrections for each of the axis on the AT telescope,
    e.g. M1 and M2 pressure and hexapod correction.

    The AOS corrections are simple polynomes where x is the gravity vector,
    cos(theta), where theta is the azimuth distance or 90.-altitude.

    """

    def __init__(self, log):

        # Create a logger if none were passed during the instantiation of
        # the class
        if log is None:
            self.log = logging.getLogger(type(self).__name__)
        else:
            self.log = log.getChild(type(self).__name__)

        self.config = {
            "m1": [0.0],
            "m2": [0.0],
            "hexapod_x": [0.0],
            "hexapod_y": [0.0],
            "hexapod_z": [0.0],
            "hexapod_u": [0.0],
            "hexapod_v": [0.0],
            "chromatic_dependence": [0.0],
        }

        self.offset = {
            "m1": 0.0,
            "m2": 0.0,
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "u": 0.0,
            "v": 0.0,
        }

        self.poly_m1 = np.poly1d(self.config["m1"])
        self.poly_m2 = np.poly1d(self.config["m2"])
        self.poly_x = np.poly1d(self.config["hexapod_x"])
        self.poly_y = np.poly1d(self.config["hexapod_y"])
        self.poly_z = np.poly1d(self.config["hexapod_z"])
        self.poly_u = np.poly1d(self.config["hexapod_u"])
        self.poly_v = np.poly1d(self.config["hexapod_v"])
        self.poly_chromatic = np.poly1d(self.config["chromatic_dependence"])

    def reset_offset(self):
        """ Reset all offsets to zero.
        """
        self.offset = {
            "m1": 0.0,
            "m2": 0.0,
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "u": 0.0,
            "v": 0.0,
        }

    def set_offset(self, axis, value):
        """ Set offset for specified axis.

        Parameters
        ----------
        axis: str
            Name of the axis. One of m1, m2, x, y, z, u, v.
        value: float
            Offset value.

        """
        if axis in self.offset:
            self.offset[axis] = float(value)
        else:
            raise RuntimeError(
                f"Invalid axis name '{axis}'. Must be one of " f"{self.offset.keys()}."
            )

    def add_offset(self, axis, value):
        """ Set offset for specified axis.

        Parameters
        ----------
        axis: str
            Name of the axis. One of m1, m2, x, y, z, u, v.
        value: float
            Offset value.

        """
        if axis in self.offset:
            self.offset[axis] += float(value)
        else:
            raise RuntimeError(
                f"Invalid axis name '{axis}'. Must be one of " f"{self.offset.keys()}."
            )

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
        return self.poly_m1(np.cos(np.radians(90.0 - elevation))) + self.offset["m1"]

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
        return self.poly_m2(np.cos(np.radians(90.0 - elevation))) + self.offset["m2"]

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
            [DISABLED] rotation angle with respect to z-axis (degrees)
        """
        x = self.poly_x(np.cos(np.radians(90.0 - elevation))) + self.offset["x"]
        y = self.poly_y(np.cos(np.radians(90.0 - elevation))) + self.offset["y"]
        z = self.poly_z(np.cos(np.radians(90.0 - elevation))) + self.offset["z"]
        u = self.poly_u(np.cos(np.radians(90.0 - elevation))) + self.offset["u"]
        v = self.poly_v(np.cos(np.radians(90.0 - elevation))) + self.offset["v"]
        w = 0.0

        return x, y, z, u, v, w

    def get_correction_chromatic(self, wavelength):
        """Focus (via z-hexapod offset) correction for specified wavelength.
        Note that the central wavelength of the telescope without any
        filter is 700nm.

        Parameters
        ----------
        wavelength : `float`
            Wavelength for the focus correction (nm).

        Returns
        -------
        z-offset : float
            z-axis focus offset (um)
        """
        _chromatic_focus_offset = self.poly_chromatic(wavelength - 700)

        self.log.debug(
            "Chromatic_focus offset is "
            f"{_chromatic_focus_offset} [mm] for wavelength {wavelength} [nm]"
        )

        return _chromatic_focus_offset

    @property
    def m1(self):
        return self.config["m1"]

    @m1.setter
    def m1(self, val):
        self.config["m1"] = val
        self.poly_m1 = np.poly1d(val)

    @property
    def m2(self):
        return self.config["m2"]

    @m2.setter
    def m2(self, val):
        self.config["m2"] = val
        self.poly_m2 = np.poly1d(val)

    @property
    def hexapod_x(self):
        return self.config["hexapod_x"]

    @hexapod_x.setter
    def hexapod_x(self, val):
        self.config["hexapod_x"] = val
        self.poly_x = np.poly1d(val)

    @property
    def hexapod_y(self):
        return self.config["hexapod_y"]

    @hexapod_y.setter
    def hexapod_y(self, val):
        self.config["hexapod_y"] = val
        self.poly_y = np.poly1d(val)

    @property
    def hexapod_z(self):
        return self.config["hexapod_z"]

    @hexapod_z.setter
    def hexapod_z(self, val):
        self.config["hexapod_z"] = val
        self.poly_z = np.poly1d(val)

    @property
    def hexapod_u(self):
        return self.config["hexapod_u"]

    @hexapod_u.setter
    def hexapod_u(self, val):
        self.config["hexapod_u"] = val
        self.poly_u = np.poly1d(val)

    @property
    def hexapod_v(self):
        return self.config["hexapod_v"]

    @hexapod_v.setter
    def hexapod_v(self, val):
        self.config["hexapod_v"] = val
        self.poly_v = np.poly1d(val)

    @property
    def chromatic_dependence(self):
        return self.config["chromatic_dependence"]

    @chromatic_dependence.setter
    def chromatic_dependence(self, val):
        self.config["chromatic_dependence"] = val
        self.poly_chromatic = np.poly1d(val)
