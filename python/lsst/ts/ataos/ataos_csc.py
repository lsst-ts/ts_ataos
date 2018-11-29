
import asyncio
import logging

import SALPY_ATAOS

from lsst.ts.salobj import base_csc

from .model import Model

__all__ = ['ATAOS']

SEEING_LOOP_DONE = 101
TELEMETRY_LOOP_DONE = 102


class ATAOS(base_csc.BaseCsc):
    """
    Commandable SAL Component (CSC) for the Auxiliary Telescope Active Optics System.
    """

    def __init__(self):
        """
        Initialize AT AOS CSC.
        """
        self.log = logging.getLogger("ATAOS")

        self.model = Model()  # instantiate the model so I can have the settings once the component is up

        super().__init__(SALPY_ATAOS)

        # set/publish summaryState
        self.summary_state = base_csc.State.STANDBY

        # publish settingVersions
        settingVersions_topic = self.evt_settingVersions.DataType()
        settingVersions_topic.recommendedSettingsVersion = \
            self.model.recommended_settings
        settingVersions_topic.recommendedSettingsLabels = self.model.settings_labels

        self.evt_settingVersions.put(settingVersions_topic)

        self.loop_die_timeout = 5  # how long to wait for the loops to die?

        self.telemetry_loop_running = False
        self.telemetry_loop_task = None

        self.health_monitor_loop_task = asyncio.ensure_future(self.health_monitor())

    def do_applyCorrection(self, id_data):
        """Apply correction.

        Parameters
        ----------
        id_data : `CommandIdData`
            Command ID and data
        """
        self.assert_enabled('applyCorrection')

    def do_applyFocusOffset(self, id_data):
        """

        Parameters
        ----------
        id_data : `CommandIdData`
            Command ID and data
        """
        self.assert_enabled('applyFocusOffset')

    def do_disableCorrection(self, id_data):
        """

        Parameters
        ----------
        id_data : `CommandIdData`
            Command ID and data
        """
        self.assert_enabled('disableCorrection')

    def do_enableCorrection(self, id_data):
        """

        Parameters
        ----------
        id_data : `CommandIdData`
            Command ID and data
        """
        self.assert_enabled('enableCorrection')

    def do_setFocus(self, id_data):
        """

        Parameters
        ----------
        id_data : `CommandIdData`
            Command ID and data
        """
        self.assert_enabled('setFocus')

    async def health_monitor(self):
        """Monitor general health of component. Transition publish `errorCode` and transition
        to FAULT state if anything bad happens.
        """
        asyncio.sleep(0)
