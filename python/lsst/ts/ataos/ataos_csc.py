
import asyncio
import logging

import SALPY_ATAOS

import SALPY_ATMCS
import SALPY_ATPneumatics
import SALPY_ATHexapod

from lsst.ts.salobj import base_csc, Remote

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

        # Remotes
        self.mcs = Remote(SALPY_ATMCS)
        self.pneumatics = Remote(SALPY_ATPneumatics)
        self.hexapod = Remote(SALPY_ATHexapod)

        # Corrections
        self.valid_corrections = ('all', 'm1', 'm2', 'hexapod', 'focus', 'moveWhileExposing')
        self.corrections = {'m1': False,
                            'm2': False,
                            'hexapod': False,
                            'focus': False,
                            }
        self.move_while_exposing = False

    async def do_applyCorrection(self, id_data):
        """Apply correction on all components either for the current position of the telescope
        (default) or the specified position.

        Since SAL does not allow definition of default parameters, azimuth = 0. and
        altitude = 0. is considered as "current telescope position".

        Angles wraps:
            altitude: (0., 90.] degrees (Model may still apply more restringing limits)
            azimuth: [0., 360.] degrees

        Parameters
        ----------
        id_data : `CommandIdData`
            Command ID and data

        Raises
        ------
        IOError
            If angles are outside bounds.
        AssertionError
            If one or more corrections are enabled.
        """
        self.assert_enabled('applyCorrection')
        self.assert_corrections('disabled')

    async def do_applyFocusOffset(self, id_data):
        """Adds a focus offset to the focus correction.

        Parameters
        ----------
        id_data : `CommandIdData`
            Command ID and data
        """
        self.assert_enabled('applyFocusOffset')

    async def do_enableCorrection(self, id_data):
        """Enable corrections on specified axis.

        This method only works for enabling features, it won't disable any of the features off
        (including `move_while_exposing`).

        Components set to False in this command won't cause any affect, e.g. if m1 correction is
        enabled and `enable_correction` receives `id_data.data.m1=False`, the correction will still be
        enabled afterwards. To disable a correction (or `move_while_exposing`) use
        `do_disableCorrection()`.

        Parameters
        ----------
        id_data : `CommandIdData`
            Command ID and data.
        """
        self.assert_enabled('enableCorrection')
        self.assert_any_corrections(id_data.data)
        asyncio.sleep(0.)  # give control back to event loop
        self.mark_corrections(id_data.data, True)
        asyncio.sleep(0.)  # give control back to event loop
        self.publish_enable_corrections()

    async def do_disableCorrection(self, id_data):
        """Disable corrections on specified axis.

        This is the mirror method of `enable_correction`, and will only disable
        features (including `move_while_exposing`).

        Parameters
        ----------
        id_data : `CommandIdData`
            Command ID and data
        """
        self.assert_enabled('disableCorrection')
        self.assert_any_corrections(id_data.data)
        asyncio.sleep(0.)  # give control back to event loop
        self.mark_corrections(id_data.data, False)
        asyncio.sleep(0.)  # give control back to event loop
        self.publish_enable_corrections()

    async def do_setFocus(self, id_data):
        """Set focus position.

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

    async def correction_loop(self):
        """Coroutine to send corrections to m1, m2, hexapod and focus."""
        pass

    def assert_any_corrections(self, data):
        """Check that at least one attribute of SALPY_ATAOS.ATAOS_command_disableCorrectionC
        or SALPY_ATAOS.ATAOS_command_enableCorrectionC are set to True.

        Parameters
        ----------
        data : SALPY_ATAOS.ATAOS_command_disableCorrectionC or SALPY_ATAOS.ATAOS_command_enableCorrectionC

        Raises
        ------
        AssertionError
            If one of more attribute of the topic is set to True.
        """
        assert any([getattr(data, corr)
                    for corr in self.valid_corrections]), \
            "At least one correction must be set."

    def assert_corrections(self, mode):
        """Check that the corrections are either enabled or disabled.

        Parameters
        ----------
        mode : str
            Either enabled or disabled

        Raises
        ------
        AssertionError
            If one or more corrections are enabled or disabled
        IOError
            If mode is not enabled or disabled
        """
        if mode not in ('enabled', 'disabled'):
            raise IOError("Mode must be either enabled or disabled")

        enabled = ''
        for key in self.corrections:
            if self.corrections[key]:
                enabled += key+','

        if mode == 'enabled':
            assert any(self.corrections.items()), "All corrections disabled"
        else:
            assert not any(self.corrections.items()), "Corrections %s enabled." % enabled

    def mark_corrections(self, data, flag):
        """Utility method to switch corrections on/off.

        Parameters
        ----------
        data : SALPY_ATAOS.ATAOS_command_disableCorrectionC or SALPY_ATAOS.ATAOS_command_enableCorrectionC
        flag : bool
        """
        if data.moveWhileExposing:
            self.move_while_exposing = flag

        if data.all:
            for key in self.corrections:
                self.corrections[key] = flag
            return

        for key in self.corrections:
            if getattr(data, key):
                self.corrections[key] = flag

    def publish_enable_corrections(self):
        """Utility function to publish enable corrections."""
        topic = self.evt_correctionEnabled.DataType()
        for key in self.corrections:
            setattr(topic, key, self.corrections[key])
        topic.moveWhileExposing = self.move_while_exposing
        self.evt_correctionEnabled.put(topic)
