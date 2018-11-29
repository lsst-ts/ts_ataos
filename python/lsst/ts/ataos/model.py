
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
