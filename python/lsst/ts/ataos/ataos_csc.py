
import asyncio
import traceback
import enum
import pathlib
import numpy as np

from lsst.ts.salobj import (base_csc, ConfigurableCsc, Remote, State,
                            AckError, SalRetCode)

from lsst.ts.idl.enums import ATPneumatics

from .model import Model

__all__ = ['ATAOS', 'ShutterState', "DetailedState"]

CORRECTION_LOOP_DIED = 8103
"""Error code for when the correction loop dies and the CSC is in enable
state.
"""


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


class ATAOS(ConfigurableCsc):
    """
    Configurable Commandable SAL Component (CSC) for the Auxiliary Telescope
    Active Optics System.
    """

    def __init__(self, config_dir=None, initial_state=State.STANDBY):
        """
        Initialize AT AOS CSC.
        """
        schema_path = pathlib.Path(__file__).resolve().parents[4].joinpath("schema", "ATAOS.yaml")

        super().__init__("ATAOS", index=0,
                         schema_path=schema_path,
                         config_dir=config_dir,
                         initial_state=initial_state,
                         initial_simulation_mode=0)

        self.model = Model()

        self._detailed_state = DetailedState.IDLE

        # how long to wait for the loops to die? = 5 heartbeats
        self.loop_die_timeout = 5.*base_csc.HEARTBEAT_INTERVAL
        # regular timeout for commands to remotes = 60 heartbeats (!?)
        self.cmd_timeout = 60.*base_csc.HEARTBEAT_INTERVAL

        # fast timeout
        self.fast_timeout = 5.*base_csc.HEARTBEAT_INTERVAL

        self.correction_loop_task = None
        # Time between corrections
        self.correction_loop_time = base_csc.HEARTBEAT_INTERVAL

        self.camera_exposing = False  # flag to monitor if camera is exposing

        # Remotes
        self.mcs = Remote(self.domain, "ATMCS", include=["target", "mount_AzEl_Encoders"])
        self.pneumatics = Remote(self.domain, "ATPneumatics", include=["m1SetPressure",
                                                                       "m2SetPressure",
                                                                       "m1OpenAirValve",
                                                                       "m2OpenAirValve",
                                                                       "m1CloseAirValve",
                                                                       "m2CloseAirValve",
                                                                       "openMasterAirSupply",
                                                                       "openInstrumentAirValve",
                                                                       "m1State",
                                                                       "m2State",
                                                                       "instrumentState",
                                                                       "mainValveState",
                                                                       "summaryState"])

        self.hexapod = Remote(self.domain, "ATHexapod", include=["moveToPosition", "positionUpdate"])
        self.camera = Remote(self.domain, "ATCamera", include=["shutterDetailedState"])

        self.pneumatics_summary_state = None
        self.pneumatics_main_valve_state = None
        self.pneumatics_instrument_valve_state = None
        self.pneumatics_m1_state = None
        self.pneumatics_m2_state = None

        self.target_azimuth = None
        self.target_elevation = None

        self.azimuth = None
        self.elevation = None

        # Add required callbacks
        self.camera.evt_shutterDetailedState.callback = self.shutter_monitor_callback

        # Corrections
        self.valid_corrections = ('enableAll', 'disableAll', 'm1', 'm2', 'hexapod', 'focus',
                                  'moveWhileExposing')

        self.corrections = {'m1': False,
                            'm2': False,
                            'hexapod': False,
                            'focus': False,
                            }

        # Note that focus is not part of corrections routines, focus correction
        # is performed by the hexapod. A different logic is used when focus
        # only correction is requested.
        self.corrections_routines = {'m1': self.set_pressure_m1,
                                     'm2': self.set_pressure_m2,
                                     'hexapod': self.set_hexapod
                                     }

        self._move_while_exposing = False

        self.log.debug("Done")

    @property
    def detailed_state(self):
        """Return the current value for detailed state.

        Returns
        -------
        detailed_state : np.uint8

        """
        return np.uint8(self._detailed_state)

    @detailed_state.setter
    def detailed_state(self, value):
        """Set and publish current value for detailed state.

        Parameters
        ----------
        value : `int`
            New detailed state. Will be converted to np.uint8
        """
        self._detailed_state = np.uint8(value)
        self.evt_detailedState.set_put(substate=self._detailed_state)

    @property
    def move_while_exposing(self):
        """Property to map the value of an attribute to the event topic."""
        # bool(self.evt_correctionEnabled.data.moveWhileExposing)
        return bool(self._move_while_exposing)

    @move_while_exposing.setter
    def move_while_exposing(self, value):
        """Set value of attribute directly to the event topic."""
        # FIXME: For some reason setting and getting straight out of the
        # topic was not working I'll leave this as a placeholder here and
        # debug this properly later.
        # self.evt_correctionEnabled.set(moveWhileExposing=bool(value))
        self._move_while_exposing = bool(value)

    async def begin_start(self, data):
        """Begin do_start; called before state changes.

        Get state information from ATPneumatics and set callbacks to monitor
        state of the component.

        Parameters
        ----------
        data : `DataType`
            Command data
        """

        if self.pneumatics_summary_state is None:
            try:
                await self.pneumatics.evt_summaryState.next(flush=False,
                                                            timeout=self.fast_timeout)
            except asyncio.TimeoutError:
                self.log.warning("Could not get summary state from ATPneumatics.")
            # Trick to get last value, until aget is available.
            ss = self.pneumatics.evt_summaryState.get()
            if ss is not None:
                self.pneumatics_summary_state = State(ss.summaryState)

            # set callback to monitor summary state from now on...
            self.pneumatics.evt_summaryState.callback = self.pneumatics_ss_callback

        if self.pneumatics_main_valve_state is None:
            try:
                await self.pneumatics.evt_mainValveState.next(flush=False,
                                                              timeout=self.fast_timeout)
            except asyncio.TimeoutError:
                self.log.warning("Could not get main valve state from ATPneumatics.")

            # Trick to get last value, until aget is available.
            mvs = self.pneumatics.evt_mainValveState.get()
            if mvs is not None:
                self.pneumatics_main_valve_state = ATPneumatics.AirValveState(mvs.state)

            # set callback to monitor main valve state from now on...
            self.pneumatics.evt_mainValveState.callback = self.pneumatics_mvs_callback

        if self.pneumatics_instrument_valve_state is None:
            try:
                await self.pneumatics.evt_instrumentState.next(flush=False,
                                                               timeout=self.fast_timeout)
            except asyncio.TimeoutError:
                self.log.warning("Could not get instrument valve state from ATPneumatics.")

            # Trick to get last value, until aget is available.
            ivs = self.pneumatics.evt_instrumentState.get()
            if ivs is not None:
                self.pneumatics_instrument_valve_state = ATPneumatics.AirValveState(ivs.state)

            # set callback to monitor instrument valve state from now on...
            self.pneumatics.evt_instrumentState.callback = self.pneumatics_iv_callback

        if self.pneumatics_m1_state is None:
            try:
                await self.pneumatics.evt_m1State.next(flush=False,
                                                       timeout=self.fast_timeout)
            except asyncio.TimeoutError:
                self.log.warning("Could not get m1 valve state from ATPneumatics.")

            # Trick to get last value, until aget is available.
            m1s = self.pneumatics.evt_m1State.get()
            if m1s is not None:
                self.pneumatics_m1_state = ATPneumatics.AirValveState(m1s.state)

            # set callback to monitor m1 valve state from now on...
            self.pneumatics.evt_m1State.callback = self.pneumatics_m1s_callback

        if self.pneumatics_m2_state is None:
            try:
                await self.pneumatics.evt_m2State.next(flush=False,
                                                       timeout=self.fast_timeout)
            except asyncio.TimeoutError:
                self.log.warning("Could not get m2 valve state from ATPneumatics.")

            # Trick to get last value, until aget is available.
            m2s = self.pneumatics.evt_m2State.get()
            if m2s is not None:
                self.pneumatics_m2_state = ATPneumatics.AirValveState(m2s.state)

            # set callback to monitor m2 valve state from now on...
            self.pneumatics.evt_m2State.callback = self.pneumatics_m2s_callback

        await super().begin_start(data)

    async def end_enable(self, id_data):
        """End do_enable; called after state changes but before command
        acknowledged.

        It will add `self.correction_loop` to the event loop.

        Parameters
        ----------
        id_data : `CommandIdData`
            Command ID and data
        """
        # Flush event queue to make sure only current values are accounted for!
        self.mcs.evt_target.flush()
        self.mcs.evt_target.callback = self.update_target_position_callback
        self.mcs.tel_mount_AzEl_Encoders.callback = self.update_mount_position_callback
        self.correction_loop_task = asyncio.ensure_future(self.correction_loop())

    async def end_disable(self, id_data):
        """End do_disable; called after state changes but before command
        acknowledged.

        Makes sure correction loop is cancelled appropriately.

        Parameters
        ----------
        id_data : `CommandIdData`
            Command ID and data
        """

        self.mcs.evt_target.callback = None
        self.target_elevation = None
        self.target_azimuth = None

        self.mcs.tel_mount_AzEl_Encoders.callback = None
        self.azimuth = None
        self.elevation = None
        if not self.correction_loop_task.done():
            self.correction_loop_task.cancel()

        try:
            await self.correction_loop_task
        except Exception as e:
            self.log.info(f"Exception while waiting for correction loop task to finish.")
            self.log.exception(e)

        disable = self.cmd_disableCorrection.DataType()
        disable.disableAll = True
        self.mark_corrections(disable, False)

        self.detailed_state = 0

    async def do_applyCorrection(self, id_data):
        """Apply correction on all components either for the current position
        of the telescope (default) or the specified position.

        Since SAL does not allow definition of default parameters,
        azimuth = 0. and altitude = 0. is considered as "current telescope
        position". Note that, if altitude > 0 and azimuth=0, the correction is
        applied at the specified position.

        Angles wraps:
            azimuth: Absolute azimuth angle. Angle will be converted to the
                0 - 360 wrap.
            elevation: (0., 90.] degrees (Model may still apply more
                restringing limits)

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
        self.assert_corrections(enabled=False)
        self.log.debug("Apply correction")

        # FIXME: Get position from telescope if elevation = 0.
        azimuth = id_data.azimuth % 360.
        elevation = id_data.elevation

        if elevation == 0.:
            if self.azimuth is None or self.elevation is None:
                raise RuntimeError("No information about telescope azimuth and/or "
                                   "elevation available.")
            # Get telescope position stored by callback function.
            azimuth = self.azimuth
            elevation = self.elevation

        self.log.debug("Apply correction Hexapod")
        await self.set_hexapod(azimuth, elevation)
        self.log.debug("Apply correction M1")
        await self.set_pressure("m1", azimuth, elevation)
        self.log.debug("Apply correction M2")
        await self.set_pressure("m2", azimuth, elevation)

        # FIXME: THIS is not working with the current version of the software
        # need to see what is the problem. 2019/June/04
        # run corrections concurrently
        # await asyncio.gather(self.set_hexapod(azimuth, elevation),
        #                      self.set_pressure("m1", azimuth, elevation),
        #                      self.set_pressure("m2", azimuth, elevation),
        #                      )  # FIXME: What about focus? YES, do focus separately

    async def do_applyFocusOffset(self, id_data):
        """Adds a focus offset to the focus correction. Same as apply focus
         but do the math...

        Parameters
        ----------
        id_data : `CommandIdData`
            Command ID and data
        """
        self.assert_enabled('applyFocusOffset')

    async def do_enableCorrection(self, id_data):
        """Enable corrections on specified axis.

        This method only works for enabling features, it won't disable any of
        the features off (including `move_while_exposing`).

        Components set to False in this command won't cause any affect, e.g.
        if m1 correction is enabled and `enable_correction` receives
        `id_data.m1=False`, the correction will still be enabled
        afterwards. To disable a correction (or `move_while_exposing`) use
        `do_disableCorrection()`.

        Parameters
        ----------
        id_data : `CommandIdData`
            Command ID and data.
        """
        self.assert_enabled('enableCorrection')
        self.assert_any_corrections(id_data)
        await asyncio.sleep(0.)  # give control back to event loop

        try:
            if id_data.m1 or id_data.enableAll:
                await self.check_atpneumatic()
                try:
                    if self.pneumatics_m1_state != ATPneumatics.AirValveState.OPENED:
                        await self.pneumatics.cmd_m1OpenAirValve.start(timeout=self.cmd_timeout)
                except AckError as e:
                    if e.ackcmd.ack == SalRetCode.CMD_NOPERM:
                        self.log.warning("M1 valve is already opened.")
                        self.log.exception(e)
                    else:
                        raise e

                # FIXME: ATPneumatics is not ready to set pressure just after
                # the valve is opened and there is currently no event to
                # indicate readiness. I'll wait 1 second after command
                # finishes. Once this is fixed the sleep can be removed.
                await asyncio.sleep(1.)

                # Set pressure to zero.
                await self.pneumatics.cmd_m1SetPressure.set_start(pressure=0.,
                                                                  timeout=self.cmd_timeout)

        except Exception as e:
            self.log.error("Failed to open m1 air valve.")
            self.log.exception(e)
            raise e

        try:
            if id_data.m2 or id_data.enableAll:
                await self.check_atpneumatic()
                try:
                    if self.pneumatics_m2_state != ATPneumatics.AirValveState.OPENED:
                        await self.pneumatics.cmd_m2OpenAirValve.start(timeout=self.cmd_timeout)
                except AckError as e:
                    if e.ackcmd.ack == SalRetCode.CMD_NOPERM:
                        self.log.warning("M2 valve is already opened.")
                        self.log.exception(e)
                    else:
                        raise e
                # Set pressure to zero.

                await self.pneumatics.cmd_m2SetPressure.set_start(pressure=0.,
                                                                  timeout=self.cmd_timeout)
        except Exception as e:
            self.log.error("Failed to open m2 air valve.")
            self.log.exception(e)
            raise e

        self.mark_corrections(id_data, True)
        await asyncio.sleep(0.)  # give control back to event loop
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
        self.assert_any_corrections(id_data)
        await asyncio.sleep(0.)  # give control back to event loop

        try:
            if id_data.m1 or id_data.disableAll:
                # Setting m1 pressure to zero and close valve
                self.pneumatics.cmd_m1SetPressure.set(pressure=0.)
                await self.pneumatics.cmd_m1SetPressure.start(timeout=self.cmd_timeout)
                await self.pneumatics.cmd_m1CloseAirValve.start(timeout=self.cmd_timeout)
        except Exception as e:
            self.log.error("Failed to close m1 air valve.")
            self.log.exception(e)

        try:
            if id_data.m2 or id_data.disableAll:
                # Setting m1 pressure to zero and close valve
                self.pneumatics.cmd_m2SetPressure.set(pressure=0.)
                await self.pneumatics.cmd_m2SetPressure.start(timeout=self.cmd_timeout)
                await self.pneumatics.cmd_m2CloseAirValve.start(timeout=self.cmd_timeout)
        except Exception as e:
            self.log.error("Failed to close m2 air valve.")
            self.log.exception(e)

        self.mark_corrections(id_data, False)
        await asyncio.sleep(0.)  # give control back to event loop
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
        """Monitor general health of component. Transition publish `errorCode`
        and transition to FAULT state if anything bad happens.
        """
        while True:
            asyncio.sleep(base_csc.HEARTBEAT_INTERVAL)

    async def correction_loop(self):
        """Coroutine to send corrections to m1, m2, hexapod and focus at
        the heartbeat frequency."""

        while self.summary_state == State.ENABLED:
            try:

                # The heartbeat wait is folded here so instead of applying and
                # then waiting, the loop applies and wait at the same time.
                corrections_to_apply = []
                if self.azimuth is not None and self.elevation is not None:
                    elevation = self.elevation
                    azimuth = self.azimuth

                    if self.target_elevation is not None and self.target_elevation < self.elevation:
                        # Telescope in going down, need to go ahead and decrease
                        # pressure accordingly
                        elevation = (self.target_elevation + self.elevation)/2.
                        self.log.debug(f"Telescope going down, getting ahead on correction."
                                       f"el: {self.elevation}, target_el: {self.target_elevation}, "
                                       f"corr_el: {elevation}")

                    for correction in self.corrections:
                        if self.corrections[correction] and correction in self.corrections_routines:
                            self.log.debug(f"Adding {correction} correction.")
                            corrections_to_apply.append(
                                self.corrections_routines[correction](azimuth,
                                                                      elevation))
                else:
                    self.log.debug("No information available about telescope azimuth and/or "
                                   "elevation.")

                # FIXME:
                # Run corrections in series because CSCs are not supporting
                # concurrent operations yet 2019/June/4
                corrections_to_apply.append(asyncio.sleep(self.correction_loop_time))
                for corr in corrections_to_apply:
                    await corr

                # run corrections concurrently (and/or wait for the heartbeat
                # interval)
                # if len(corrections_to_apply) > 0:
                #     await asyncio.gather(*corrections_to_apply)

                # await asyncio.sleep(self.correction_loop_time)

            except asyncio.CancelledError:
                self.log.debug("Correction loop cancelled.")
                break
            except Exception as e:
                self.log.error("Error in correction loop. Going to FAULT state.")
                self.log.exception(e)
                self.evt_errorCode.set_put(errorCode=CORRECTION_LOOP_DIED,
                                           errorReport="Correction loop died.",
                                           traceback=traceback.format_exc())
                self.fault()
                break

    def assert_any_corrections(self, data):
        """Check that at least one attribute of
        SALPY_ATAOS.ATAOS_command_disableCorrectionC or
        SALPY_ATAOS.ATAOS_command_enableCorrectionC are set to True.

        Parameters
        ----------
        data : SALPY_ATAOS.ATAOS_command_disableCorrectionC or
               SALPY_ATAOS.ATAOS_command_enableCorrectionC

        Raises
        ------
        AssertionError
            If one of more attribute of the topic is set to True.
        """

        assert any([getattr(data, corr, False)
                    for corr in self.valid_corrections]), \
            "At least one correction must be set."

    def assert_corrections(self, enabled):
        """Check that the corrections are either enabled or disabled.

        Parameters
        ----------
        enabled : bool
            Specify if a correction is enabled (True) or disabled (False).

        Raises
        ------
        AssertionError
            If one or more corrections are enabled or disabled
        IOError
            If mode is not enabled or disabled
        """

        if enabled:
            assert any(self.corrections.values()), "All corrections disabled"
        else:
            enabled_keys = [key for key, is_enabled in self.corrections.items() if is_enabled]
            assert not any(self.corrections.values()), \
                "Corrections %s enabled: %s." % (enabled_keys, self.corrections.items())

    def can_move(self):
        """Check that it is ok to move the hexapod.

        If `self.move_while_exposing = False` the method will check that
        the camera is not exposing. If it is exposing, return `False`.
        Return `True` otherwise. To determine if the camera is exposing or not
        it uses a callback to the ATCamera_logevent_shutterDetailedState.

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
        data : SALPY_ATAOS.ATAOS_command_disableCorrectionC or
               SALPY_ATAOS.ATAOS_command_enableCorrectionC
        flag : bool
        """
        if data.moveWhileExposing:
            self.move_while_exposing = flag
        # Note that ATAOS_command_disableCorrectionC and
        # ATAOS_command_enableCorrectionC topics have different names for the
        # attribute that acts on all axis. They could have the same name but I
        # wanted to make sure it is clear that when someone uses
        # ATAOS_command_disableCorrectionC.disableAll they really understand
        # they are disabling all corrections and vice-versa. With the
        # different names I'm forced to do the following getattr logic.
        # Either that or check the type of the input. But I think this looks
        # cleaner.
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
        self.evt_correctionEnabled.set_put(moveWhileExposing=self.move_while_exposing,
                                           **kwargs)

    def shutter_monitor_callback(self, data):
        """A callback function to monitor the camera shutter.

        Parameters
        ----------
        data : `SALPY_ATCamera.ATCamera_logevent_shutterDetailedStateC`
            Command ID and data
        """
        self.camera_exposing = data.substate != ShutterState.CLOSED

    async def set_pressure_m1(self, azimuth, elevation):
        """Set pressure on m1.

        Parameters
        ----------
        azimuth : float
        elevation : float
        """
        await self.set_pressure("m1", azimuth, elevation)

    async def set_pressure_m2(self, azimuth, elevation):
        """Set pressure on m2.

        Parameters
        ----------
        azimuth : float
        elevation : float
        """
        await self.set_pressure("m2", azimuth, elevation)

    async def set_pressure(self, mirror, azimuth, elevation):
        """Set pressure on specified mirror.

        Parameters
        ----------
        mirror : str
            Either m1 or m2
        azimuth : float
        elevation : float
        """
        status_bit = getattr(DetailedState, f"{mirror}".upper())

        # Check that pressure is not being applied yet
        if self.detailed_state & status_bit != 0:
            self.log.warning("%s pressure correction running... skipping...", mirror)
            return
        else:
            # publish new detailed state
            self.detailed_state = self.detailed_state ^ status_bit

            cmd_attr = getattr(self.pneumatics, f"cmd_{mirror}SetPressure")
            evt_start_attr = getattr(self, f"evt_{mirror}CorrectionStarted")
            evt_end_attr = getattr(self, f"evt_{mirror}CorrectionCompleted")

            cmd_topic = cmd_attr.DataType()
            cmd_topic.pressure = getattr(self.model, f"get_correction_{mirror}")(azimuth,
                                                                                 elevation)

            await asyncio.sleep(0.)  # give control back to the event loop

            start_topic = evt_start_attr.DataType()
            start_topic.azimuth = azimuth
            start_topic.elevation = elevation
            start_topic.pressure = cmd_topic.pressure

            end_topic = evt_end_attr.DataType()
            end_topic.azimuth = azimuth
            end_topic.elevation = elevation
            end_topic.pressure = cmd_topic.pressure

            evt_start_attr.put(start_topic)
            try:
                await cmd_attr.start(cmd_topic,
                                     timeout=self.cmd_timeout)
            except Exception as e:
                self.log.warning(f"Failed to set pressure for {mirror} @ "
                                 f"AzEl: {azimuth}/{elevation}")
                self.log.exception(e)
                raise e
            finally:
                evt_end_attr.put(end_topic)
                # correction completed... flip bit on detailedState
                self.detailed_state = self.detailed_state ^ status_bit

    async def set_focus(self, azimuth, elevation):
        """Utility to set focus position.

        Parameters
        ----------
        azimuth
        elevation
        """
        await self.set_hexapod(azimuth, elevation, f'z')

    async def set_hexapod(self, azimuth, elevation, axis=f'xyzuvw'):
        """Utility to set hexapod position.

        Parameters
        ----------
        azimuth
        elevation
        axis
        """

        if self.can_move():
            # publish new detailed state

            self.log.debug(f"Moving hexapod axis {axis}")

            status_bit = DetailedState.HEXAPOD
            cmd_attr = getattr(self.hexapod, f"cmd_moveToPosition")
            evt_start_attr = getattr(self, f"evt_hexapodCorrectionStarted")
            evt_end_attr = getattr(self, f"evt_hexapodCorrectionCompleted")

            if axis == f"z":
                status_bit = DetailedState.FOCUS
                evt_start_attr = getattr(self, f"evt_focusCorrectionStarted")
                evt_end_attr = getattr(self, f"evt_focusCorrectionCompleted")

            self.detailed_state = self.detailed_state ^ status_bit

            cmd_topic = cmd_attr.DataType()
            (cmd_topic.x, cmd_topic.y, cmd_topic.z,
             cmd_topic.u, cmd_topic.v, cmd_topic.w) = self.model.get_correction_hexapod(azimuth,
                                                                                        elevation)

            start_topic = evt_start_attr.DataType()
            start_topic.azimuth = azimuth
            start_topic.elevation = elevation

            end_topic = evt_end_attr.DataType()
            end_topic.azimuth = azimuth
            end_topic.elevation = elevation

            for ax in axis:
                if axis == f"z":
                    setattr(start_topic, f'focus', getattr(cmd_topic, ax))
                    setattr(end_topic, f'focus', getattr(cmd_topic, ax))
                else:
                    setattr(start_topic, f'hexapod_{ax}', getattr(cmd_topic, ax))
                    setattr(end_topic, f'hexapod_{ax}', getattr(cmd_topic, ax))

            evt_start_attr.put(start_topic)
            try:
                coro = self.hexapod.evt_positionUpdate.next(flush=True, timeout=self.cmd_timeout)
                await cmd_attr.start(cmd_topic,
                                     timeout=self.cmd_timeout)
                await coro
            except Exception as e:
                self.log.warning(f"Failed to set hexapod position @ "
                                 f"AzEl: {azimuth}/{elevation}")
                self.log.exception(e)
                raise e
            finally:
                evt_end_attr.put(end_topic)
                # correction completed... flip bit on detailedState
                self.detailed_state = self.detailed_state ^ status_bit

    def update_target_position_callback(self, id_data):
        """Callback function to update the target telescope position.

        Parameters
        ----------
        id_data : SALPY_ATMCS.ATMCS_logevent_target

        """
        self.target_azimuth = id_data.azimuth
        self.target_elevation = id_data.elevation

    def update_mount_position_callback(self, id_data):
        """Callback function to update the position of the telescope.

        Parameters
        ----------
        id_data : SALPY_ATMCS.ATMCS_mountEncoders

        """
        self.azimuth = id_data.azimuthCalculatedAngle[-1]
        self.elevation = id_data.elevationCalculatedAngle[-1]

    @staticmethod
    def get_config_pkg():
        return "ts_config_attcs"

    async def configure(self, config):
        """Implement method to configure the CSC.

        Parameters
        ----------
        config : `object`
            The configuration as described by the schema at ``schema_path``,
            as a struct-like object.
        Notes
        -----
        Called when running the ``start`` command, just before changing
        summary state from `State.STANDBY` to `State.DISABLED`.

        """

        self.correction_loop_time = 1./config.correction_frequency

        for key in ['m1', 'm2', 'hexapod_x', 'hexapod_y', 'hexapod_z', 'hexapod_u', 'hexapod_v']:
            if hasattr(config, key):
                setattr(self.model, key, getattr(config, key))
            else:
                setattr(self.model, key, [0.])

    async def check_atpneumatic(self):
        """Check that the main and instrument valves on ATPneumatics are open.
        Open then is they are closed.
        """

        if self.pneumatics_summary_state != State.ENABLED:
            raise RuntimeError(f"ATPneumatics in {self.pneumatics_summary_state}. "
                               f"Expected {State.ENABLED}. Enable CSC before "
                               f"activating corrections.")

        if self.pneumatics_main_valve_state != ATPneumatics.AirValveState.OPENED:
            self.log.debug("ATPneumatics main valve not opened, trying to open it.")
            await self.pneumatics.cmd_openMasterAirSupply.start(timeout=self.cmd_timeout)

        if self.pneumatics_instrument_valve_state != ATPneumatics.AirValveState.OPENED:
            self.log.debug("ATPneumatics instrument valve not opened, trying to open it.")
            await self.pneumatics.cmd_openInstrumentAirValve.start(timeout=self.cmd_timeout)

    def fault(self, code=None, report=""):
        """Enter the fault state.

        Subclass parent method to disable corrections in the wait to FAULT
        state.

        Parameters
        ----------
        code : `int` (optional)
            Error code for the ``errorCode`` event; if None then ``errorCode``
            is not output and you should output it yourself.
        report : `str` (optional)
            Description of the error.
        """

        # Disable corrections
        self.mcs.evt_target.callback = None
        disable_corr = self.cmd_disableCorrection.DataType()
        disable_corr.disableAll = True
        self.mark_corrections(disable_corr, False)

        self.publish_enable_corrections()

        super().fault(code=code, report=report)

    def pneumatics_ss_callback(self, data):
        """Callback to monitor summary state from atpneumatics.

        Parameters
        ----------
        data

        """
        self.pneumatics_summary_state = State(data.summaryState)

    def pneumatics_mvs_callback(self, data):
        """Callback to monitor main valve state from atpneumatics.

        Parameters
        ----------
        data

        """
        self.pneumatics_main_valve_state = ATPneumatics.AirValveState(data.state)

    def pneumatics_iv_callback(self, data):
        """Callback to monitor instrument valve state from atpneumatics.

        Parameters
        ----------
        data

        """
        self.pneumatics_instrument_valve_state = ATPneumatics.AirValveState(data.state)

    def pneumatics_m1s_callback(self, data):
        """Callback to monitor m1 valve state from atpneumatics.

        Parameters
        ----------
        data

        """
        self.pneumatics_m1_state = ATPneumatics.AirValveState(data.state)

    def pneumatics_m2s_callback(self, data):
        """Callback to monitor m2 valve state from atpneumatics.

        Parameters
        ----------
        data

        """
        self.pneumatics_m2_state = ATPneumatics.AirValveState(data.state)
