
__all__ = ['Model']


class Model:

    @property
    def recommended_settings(self):
        """Recommended settings property.

        Returns
        -------
        recommended_settings : str
            Recommended settings read from Model configuration file.
        """
        return 'default'  # FIXME: Read from config file

    @property
    def settings_labels(self):
        """Recommended settings labels.

        Returns
        -------
        recommended_settings_labels : str
            Recommended settings labels read from Model configuration file.

        """
        return 'default,option1,test'  # FIXME: Read from config file

    def get_correction_m1(self, azimuth, elevation):
        """

        Parameters
        ----------
        azimuth
        elevation

        Returns
        -------
        pressure : float
            Pressure to apply (Pascal).
        """
        return 0.

    def get_correction_m2(self, azimuth, elevation):
        """

        Parameters
        ----------
        azimuth
        elevation

        Returns
        -------
        pressure : float
            Pressure to apply (Pascal).
        """
        return 0.

    def get_correction_hexapod(self, azimuth, elevation):
        """

        Parameters
        ----------
        azimuth
        elevation

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
        return 0., 0., 0., 0., 0., 0.
