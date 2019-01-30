
import asyncio
import logging
import enum
import numpy as np

import SALPY_ATAOS

import SALPY_ATMCS
import SALPY_ATPneumatics
import SALPY_ATHexapod
import SALPY_ATCamera

from lsst.ts.salobj import base_csc, Remote

from .model import Model

__all__ = ['ATAOS', 'ShutterState']

SEEING_LOOP_DONE = 101
TELEMETRY_LOOP_DONE = 102


class ShutterState(enum.IntEnum):
    """State constants.

    The numeric values come from
    https://confluence.lsstcorp.org/display/SYSENG/SAL+constraints+and+recommendations
    """
    CLOSED = 1
    OPEN = 2
    CLOSING = 3
    OPENING = 4


class DetailedState(enum.IntEnum):
    """Detailed state of the ATAOS system. The topic does not use an enumeration
    but rather an uint8 type, each byte stores information about a specific action.

    The numeric values come from
    https://confluence.lsstcorp.org/display/SYSENG/SAL+constraints+and+recommendations
    """
    IDLE = 0
    M1 = 1 << 1  # Correction to M1 pressure running
    M2 = 1 << 2  # Correction to M2 pressure running
    HEXAPOD = 1 << 3  # Hexapod correction running
    FOCUS = 1 << 4  # Focus correction running


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

        self._detailed_state = DetailedState.IDLE

        # publish settingVersions
        settingVersions_topic = self.evt_settingVersions.DataType()
        settingVersions_topic.recommendedSettingsVersion = \
            self.model.recommended_settings
        settingVersions_topic.recommendedSettingsLabels = self.model.settings_labels

        self.evt_settingVersions.put(settingVersions_topic)

        # how long to wait for the loops to die? = 5 heartbeats
        self.loop_die_timeout = 5.*base_csc.HEARTBEAT_INTERVAL
        # regular timeout for commands to remotes = 5 heartbeats
        self.cmd_timeout = 5.*base_csc.HEARTBEAT_INTERVAL

        self.telemetry_loop_running = False
        self.telemetry_loop_task = None

        self.health_monitor_loop_task = asyncio.ensure_future(self.health_monitor())

        self.camera_exposing = False  # flag to monitor if camera is exposing

        # Remotes
        self.mcs = Remote(SALPY_ATMCS)
        self.pneumatics = Remote(SALPY_ATPneumatics)
        self.hexapod = Remote(SALPY_ATHexapod)
        self.camera = Remote(SALPY_ATCamera)

        # Add required callbacks
        self.camera.evt_shutterDetailedState.callback = self.shutter_monitor_callback

        # Corrections
        self.valid_corrections = ('enableAll', 'disableAll', 'm1', 'm2', 'hexapod', 'focus',
                                  'moveWhileExposing')

        # TODO: ADD separate flag for pressure on m1 and m2 separately...
        self.corrections = {'m1': False,
                            'm2': False,
                            'hexapod': False,
                            'focus': False,
                            }

    @property
    def detailed_state(self):
        """

        Returns
        -------
        detailed_state : np.uint8

        """
        return np.uint8(self._detailed_state)

    @detailed_state.setter
    def detailed_state(self, value):
        """

        Parameters
        ----------
        value
        """
        self._detailed_state = np.uint8(value)

    @property
    def move_while_exposing(self):
        """Property to map the value of an attribute to the event topic."""
        return self.evt_correctionEnabled.data.moveWhileExposing

    @move_while_exposing.setter
    def move_while_exposing(self, value):
        """Set value of attribute directly to the event topic."""
        self.evt_correctionEnabled.data.moveWhileExposing = value

    async def do_applyCorrection(self, id_data):
        """Apply correction on all components either for the current position of the telescope
        (default) or the specified position.

        Since SAL does not allow definition of default parameters, azimuth = 0. and
        altitude = 0. is considered as "current telescope position". Note that, if altitude > 0
        and azimuth=0, the correction is applied at the specified position.

        Angles wraps:
            azimuth: [0., 360.] degrees
            elevation: (0., 90.] degrees (Model may still apply more restringing limits)

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
        self.assert_corrections(0)

        # FIXME: Get position from telescope if elevation = 0.
        azimuth = id_data.data.azimuth
        elevation = id_data.data.elevation

        if elevation == 0.:
            # Get telescope position. Will flush data stream in order to get the most updated position
            position = await self.mcs.tel_mountEncoders.next(flush=True,
                                                             timeout=self.cmd_timeout)
            azimuth = position.azimuthCalculatedAngle
            elevation = position.elevationCalculatedAngle

        # run corrections concurrently
        await asyncio.gather(self.set_hexapod(azimuth, elevation),
                             self.set_pressure("m1", azimuth, elevation),
                             self.set_pressure("m2", azimuth, elevation),
                             )  # FIXME: What about focus? YES, do focus separately

    async def do_applyFocusOffset(self, id_data):
        """Adds a focus offset to the focus correction. Same as apply focus but do the math...

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

        assert any([getattr(data, corr, False)
                    for corr in self.corrections]), \
            "At least one correction must be set."

    def assert_corrections(self, mode):
        """Check that the corrections are either enabled or disabled.

        Parameters
        ----------
        mode : int
            Specify if a correction is disabled (0) or enabled (1).

        Raises
        ------
        AssertionError
            If one or more corrections are enabled or disabled
        IOError
            If mode is not enabled or disabled
        """
        if mode > 1:
            raise ValueError("Mode must be either 0 (disabled) or 1 (enabled).")

        if mode == 1:
            assert any(self.corrections.values()), "All corrections disabled"
        else:
            enabled_keys = [key for key, is_enabled in self.corrections.items() if is_enabled]
            assert not any(self.corrections.values()), \
                "Corrections %s enabled: %s." % (enabled_keys, self.corrections.items())

    def can_move(self):
        """Check that it is ok to move the hexapod.

        If `self.move_while_exposing = False` the method will check that
        the camera is not exposing. If it is exposing, return `False`.
        Return `True` otherwise. To determine if the camera is exposing
        or not it uses a callback to the ATCamera_logevent_shutterDetailedState.
        If the detailed state is CLOSED, it means it is ok to expose.

        Returns
        -------
        can_move : bool
            True if it is ok to move the hexapod, False otherwise.

        """
        if self.move_while_exposing:  # The easy case
            return True
        elif self.camera_exposing:
            return False
        else:
            return True

    def mark_corrections(self, data, flag):
        """Utility method to switch corrections on/off.

        Parameters
        ----------
        data : SALPY_ATAOS.ATAOS_command_disableCorrectionC or SALPY_ATAOS.ATAOS_command_enableCorrectionC
        flag : bool
        """
        if data.moveWhileExposing:
            self.move_while_exposing = flag

        if getattr(data, "enableAll", False):
            for key in self.corrections:
                self.corrections[key] = flag
            return
        elif getattr(data, "disableAll", False):
            for key in self.corrections:
                self.corrections[key] = flag
            return

        for key in self.corrections:
            if getattr(data, key):
                self.corrections[key] = flag

    def publish_enable_corrections(self):
        """Utility function to publish enable corrections."""
        kwargs = dict((key, value) for key, value in self.corrections.items())
        self.evt_correctionEnabled.set_put(**kwargs)

    def shutter_monitor_callback(self, data):
        """A callback function to monitor the camera shutter.

        Parameters
        ----------
        data : `SALPY_ATCamera.ATCamera_logevent_shutterDetailedStateC`
            Command ID and data
        """
        self.camera_exposing = data.substate != ShutterState.CLOSED

    async def set_pressure(self, mirror, azimuth, elevation):
        """

        Parameters
        ----------
        mirror : str
            Either m1 or m2
        azimuth : float
        elevation : float
        """
        status_bit = getattr(DetailedState, f"{mirror}".upper())

        # Check that pressure is not being applied yet
        if self.detailed_state & (1 << status_bit) != 0:
            self.log.warning("%s pressure correction running... skipping...", mirror)
            return
        else:
            # publish new detailed state
            self.detailed_state = self.detailed_state ^ (1 << status_bit)
            detailed_state_attr = getattr(self, f"evt_detailedState")
            topic = detailed_state_attr.DataType()
            topic.substate = self.detailed_state
            detailed_state_attr.put(topic)

            cmd_attr = getattr(self.pneumatics, f"cmd_{mirror}SetPressure")
            evt_start_attr = getattr(self, f"evt_{mirror}CorrectionStarted")
            evt_end_attr = getattr(self, f"evt_{mirror}CorrectionCompleted")

            cmd_topic = cmd_attr.DataType()
            cmd_topic.pressure = getattr(self.model, f"get_correction_{mirror}")(azimuth,
                                                                                 elevation)

            asyncio.sleep(0.)  # give control back to the event loop

            start_topic = evt_start_attr.DataType()
            start_topic.azimuth = azimuth
            start_topic.elevation = elevation
            start_topic.pressure = cmd_topic.pressure

            end_topic = evt_end_attr.DataType()
            end_topic.azimuth = azimuth
            end_topic.elevation = elevation
            end_topic.pressure = cmd_topic.pressure

            evt_start_attr.put(start_topic)
            await cmd_attr.start(cmd_topic,
                                 timeout=self.cmd_timeout)
            evt_end_attr.put(end_topic)
            # correction completed... flip bit on detailedState
            self.detailed_state = self.detailed_state ^ (1 << status_bit)
            topic.substate = self.detailed_state
            detailed_state_attr.put(topic)

    async def set_hexapod(self, azimuth, elevation):
        """

        Parameters
        ----------
        azimuth
        elevation
        """
        status_bit = DetailedState.HEXAPOD

        if self.can_move():
            # publish new detailed state
            self.detailed_state = self.detailed_state ^ (1 << status_bit)
            detailed_state_attr = getattr(self, f"evt_detailedState")
            topic = detailed_state_attr.DataType()
            topic.substate = self.detailed_state
            detailed_state_attr.put(topic)

            axis = f'xyzuvw'

            cmd_attr = getattr(self.hexapod, f"cmd_moveToPosition")
            evt_start_attr = getattr(self, f"evt_hexapodCorrectionStarted")
            evt_end_attr = getattr(self, f"evt_hexapodCorrectionCompleted")

            cmd_topic = cmd_attr.DataType()
            (cmd_topic.x, cmd_topic.y, cmd_topic.z,
             cmd_topic.u, cmd_topic.v, cmd_topic.w) = self.model.get_correction_hexapod(azimuth,
                                                                                        elevation)

            asyncio.sleep(0.)  # give control back to the event loop

            start_topic = evt_start_attr.DataType()
            start_topic.azimuth = azimuth
            start_topic.elevation = elevation

            end_topic = evt_end_attr.DataType()
            end_topic.azimuth = azimuth
            end_topic.elevation = elevation

            for ax in axis:
                setattr(start_topic, f'hexapod_{ax}', getattr(cmd_topic, ax))
                setattr(end_topic, f'hexapod_{ax}', getattr(cmd_topic, ax))

            evt_start_attr.put(start_topic)
            await cmd_attr.start(cmd_topic,
                                 timeout=self.cmd_timeout)
            evt_end_attr.put(end_topic)
            # correction completed... flip bit on detailedState
            self.detailed_state = self.detailed_state ^ (1 << status_bit)
            topic.substate = self.detailed_state
            detailed_state_attr.put(topic)
