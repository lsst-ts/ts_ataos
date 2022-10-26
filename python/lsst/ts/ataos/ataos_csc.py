import asyncio
import copy
import enum
import traceback
import typing

import numpy as np
from lsst.ts.idl.enums import ATPneumatics
from lsst.ts.observatory.control.auxtel import ATCS, ATCSUsages
from lsst.ts.salobj import (
    AckError,
    ConfigurableCsc,
    Remote,
    SalRetCode,
    State,
    type_hints,
)

from lsst.ts import utils

from . import __version__
from .config_schema import CONFIG_SCHEMA
from .model import Model

__all__ = ["ATAOS", "ShutterState", "DetailedState", "run_ataos_csc"]

CORRECTION_LOOP_DIED = 8103
"""Error code for when the correction loop dies and the CSC is in enable
state.
"""


def run_ataos_csc() -> None:
    asyncio.run(ATAOS.amain(index=None))


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
    """Detailed state of the ATAOS system. The topic does not use an
    enumeration but rather an uint8 type, each byte stores information about
    a specific action.

    The numeric values come from
    https://confluence.lsstcorp.org/display/SYSENG/SAL+constraints+and+recommendations
    """

    IDLE = np.uint8(0)
    M1 = np.uint8(1 << 1)  # Correction to M1 pressure running
    M2 = np.uint8(1 << 2)  # Correction to M2 pressure running
    HEXAPOD = np.uint8(1 << 3)  # Hexapod correction running
    FOCUS = np.uint8(1 << 4)  # Focus correction running -- NOT YET IMPLEMENTED
    ATSPECTROGRAPH = np.uint8(1 << 5)  # Pointing/Focus correction running


class ATAOS(ConfigurableCsc):
    """
    Configurable Commandable SAL Component (CSC) for the Auxiliary Telescope
    Active Optics System.
    """

    valid_simulation_modes = (0,)
    version = __version__

    def __init__(
        self,
        config_dir: typing.Optional[str] = None,
        initial_state: State = State.STANDBY,
    ) -> None:
        """
        Initialize AT AOS CSC.
        """

        super().__init__(
            "ATAOS",
            index=0,
            config_schema=CONFIG_SCHEMA,
            config_dir=config_dir,
            initial_state=initial_state,
        )

        self.model = Model(self.log)

        self._detailed_state = np.uint8(DetailedState.IDLE)

        # how long to wait for the loops to die? = 5 heartbeats
        self.loop_die_timeout = 5.0 * self.heartbeat_interval
        # regular timeout for commands to remotes = 60 heartbeats (!?)
        self.cmd_timeout = 60.0 * self.heartbeat_interval

        # fast timeout
        self.fast_timeout = 5.0 * self.heartbeat_interval

        # time it takes to finish start command
        self.start_timeout = 10 * self.fast_timeout

        # Declare contents related to the asyncio lock
        self.correction_loop_lock = asyncio.Lock()
        self.correction_loop_task = utils.make_done_future()

        # Create an Event object that will get set after each
        # loop iteration
        self.correction_loop_completed_evt = asyncio.Event()

        # Time between corrections
        self.correction_loop_time = self.heartbeat_interval

        self.camera_exposing = False  # flag to monitor if camera is exposing

        # Create Remotes
        self.camera = Remote(self.domain, "ATCamera", include=["shutterDetailedState"])
        self.atspectrograph = Remote(
            self.domain,
            "ATSpectrograph",
            include=[
                "summaryState",
                "reportedFilterPosition",
                "reportedDisperserPosition",
            ],
        )

        self.atcs = ATCS(
            domain=self.domain, intended_usage=ATCSUsages.OffsettingForATAOS
        )

        self.pneumatics_summary_state = None
        self.pneumatics_main_valve_state = None
        self.pneumatics_instrument_valve_state = None
        self.pneumatics_m1_state = None
        self.pneumatics_m2_state = None

        self.atspectrograph_summary_state = None
        self.current_atspectrograph_filter_name = None
        self.current_atspectrograph_disperser_name = None
        self.current_atspectrograph_central_wavelength = None
        self.focus_offset_yet_to_be_applied = 0.0
        self.pointing_offsets_yet_to_be_applied = np.zeros((2))  # in arcsec

        # Counter for how many loops through spectrograph correction are
        # required. This is needed to satisfy race conditions that are not
        # accounted for if using a boolean. This originates from the ATAOS
        # always needing to publish events for filter/grating/wavelength
        # changes, even when no actual change is required.
        self.atspectrograph_corrections_required = 0

        # Add required callbacks for ATCamera
        self.camera.evt_shutterDetailedState.callback = self.shutter_monitor_callback

        # Corrections
        self.valid_corrections = (
            "enableAll",
            "disableAll",
            "m1",
            "m2",
            "hexapod",
            "focus",
            "atspectrograph",
            "moveWhileExposing",
        )

        self.corrections = {
            "m1": False,
            "m2": False,
            "hexapod": False,
            "focus": False,
            "atspectrograph": False,
        }

        self.current_positions = {
            "m1": None,
            "m2": None,
            "x": None,
            "y": None,
            "z": None,
            "u": None,
            "v": None,
        }

        self.focus_offset_per_category = {
            "total": 0.0,
            "userApplied": 0.0,
            "filter": 0.0,
            "disperser": 0.0,
            "wavelength": 0.0,
        }

        self.pointing_offsets_per_category = {
            "total": np.zeros((2)),
            "filter": np.zeros((2)),
            "disperser": np.zeros((2)),
        }

        # Add callback to get positions
        self.hexapod.evt_positionUpdate.callback = self.hexapod_monitor_callback
        self.pneumatics.tel_m1AirPressure.callback = self.m1_pressure_monitor_callback
        self.pneumatics.tel_m2AirPressure.callback = self.m2_pressure_monitor_callback

        self.correction_tolerance = {
            "m1": 0.0,
            "m2": 0.0,
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "u": 0.0,
            "v": 0.0,
        }

        # Note that focus is not part of corrections routines, focus correction
        # as a function of azimuth/elevation/temperature is performed by the
        # hexapod correction.
        # A different logic is used when focus only correction is requested.
        # Correction for the atspectrograph setup is only performed when an
        # event is received from ATSpectrograph saying the filter and/or
        # grating has changed.

        self.corrections_routines: typing.Dict[
            str, typing.Callable[[float, float], typing.Awaitable[None]]
        ] = {
            "m1": self.set_pressure_m1,
            "m2": self.set_pressure_m2,
            "hexapod": self.set_hexapod,
            "atspectrograph": self.set_atspectrograph_corrections,
        }

        self._move_while_exposing = False

        self.log.debug("Done __init__")

    # Create a property for all remotes used inside the atcs so the syntax
    # used to call each remote is the same
    # note that all remotes in the atcs are lowercase
    @property
    def pneumatics(self) -> Remote:
        return self.atcs.rem.atpneumatics

    @property
    def mcs(self) -> Remote:
        return self.atcs.rem.atmcs

    @property
    def hexapod(self) -> Remote:
        return self.atcs.rem.athexapod

    # create properties for azimuth/elevation aspects so that callbacks are
    # not required. These callbacks cause issues inside the ATCS class
    @property
    def azimuth(self) -> typing.Optional[float]:
        if self.atcs.telescope_position is None:
            return None
        else:
            return self.atcs.telescope_position.azimuthCalculatedAngle[-1]

    @property
    def elevation(self) -> typing.Optional[float]:
        if self.atcs.telescope_position is None:
            return None
        else:
            return self.atcs.telescope_position.elevationCalculatedAngle[-1]

    @property
    def target_azimuth(self) -> typing.Optional[float]:
        if self.atcs.telescope_target is None:
            return None
        else:
            return self.atcs.telescope_target.azimuth

    @property
    def target_elevation(self) -> typing.Optional[float]:
        if self.atcs.telescope_target is None:
            return None
        else:
            return self.atcs.telescope_target.elevation

    @property
    def detailed_state(self) -> np.uint8:
        """Return the current value for detailed state.

        Returns
        -------
        detailed_state : np.uint8

        """
        return np.uint8(self._detailed_state)

    @property
    def enable_correction_timeout(self) -> float:
        """Timeout for enableCorrection command, based on the number of
        commands sent and events waited.
        """
        return self.cmd_timeout * 6

    @property
    def apply_correction_timeout(self) -> float:
        """Timeout for applyCorrection command, based on the number of
        commands sent and events waited.
        """
        return self.cmd_timeout

    async def set_detailed_state(self, detailed_state: np.uint8) -> None:
        """Set and publish current value for detailed state.

        Parameters
        ----------
        detailed_state : `int`
            New detailed state. Will be converted to np.uint8
        """
        self._detailed_state = np.uint8(detailed_state)
        await self.evt_detailedState.set_write(substate=self._detailed_state)

    @property
    def move_while_exposing(self) -> bool:
        """Property to map the value of an attribute to the event topic."""
        # bool(self.evt_correctionEnabled.data.moveWhileExposing)
        return bool(self._move_while_exposing)

    @move_while_exposing.setter
    def move_while_exposing(self, value: bool) -> None:
        """Set value of attribute directly to the event topic."""
        # FIXME: For some reason setting and getting straight out of the
        # topic was not working I'll leave this as a placeholder here and
        # debug this properly later.
        # self.evt_correctionEnabled.set(moveWhileExposing=bool(value))
        self._move_while_exposing = bool(value)

    async def begin_start(self, data: type_hints.BaseMsgType) -> None:
        """Begin do_start; called before state changes.

        Get state information from ATPneumatics and set callbacks to monitor
        state of the component.

        Parameters
        ----------
        data : `cmd_start.DataType`
            Command data

        """
        await self.cmd_start.ack_in_progress(data, timeout=self.start_timeout)

        # Populate summary states and create callbacks (unless they
        # have already been created)
        if (
            self.pneumatics_summary_state is None
            and self.pneumatics.evt_summaryState.has_callback is False
        ):
            try:
                self.pneumatics_summary_state = (
                    await self.pneumatics.evt_summaryState.aget(
                        timeout=self.fast_timeout
                    )
                ).summaryState
            except asyncio.TimeoutError:
                self.log.warning("Could not get summary state from ATPneumatics.")

            # set callback to monitor summary state from now on...
            self.pneumatics.evt_summaryState.callback = (
                self.pneumatics_summary_state_callback
            )

        if (
            self.pneumatics_main_valve_state is None
            and self.pneumatics.evt_mainValveState.has_callback is False
        ):
            try:
                self.pneumatics_main_valve_state = (
                    await self.pneumatics.evt_mainValveState.aget(
                        timeout=self.fast_timeout
                    )
                ).state
            except asyncio.TimeoutError:
                self.log.warning("Could not get main valve state from ATPneumatics.")

            # set callback to monitor main valve state from now on...
            self.pneumatics.evt_mainValveState.callback = (
                self.pneumatics_main_valve_state_callback
            )

        if (
            self.pneumatics_instrument_valve_state is None
            and self.pneumatics.evt_instrumentState.has_callback is False
        ):
            try:
                self.pneumatics_instrument_valve_state = (
                    await self.pneumatics.evt_instrumentState.aget(
                        timeout=self.fast_timeout
                    )
                ).state
            except asyncio.TimeoutError:
                self.log.warning(
                    "Could not get instrument valve state from ATPneumatics."
                )

            # set callback to monitor instrument valve state from now on...
            self.pneumatics.evt_instrumentState.callback = (
                self.pneumatics_instrument_state_callback
            )

        if (
            self.pneumatics_m1_state is None
            and self.pneumatics.evt_m1State.has_callback is False
        ):
            try:
                self.pneumatics_m1_state = (
                    await self.pneumatics.evt_m1State.aget(timeout=self.fast_timeout)
                ).state
            except asyncio.TimeoutError:
                self.log.warning("Could not get m1 valve state from ATPneumatics.")

            # set callback to monitor m1 valve state from now on...
            self.pneumatics.evt_m1State.callback = self.pneumatics_m1_state_callback

        if (
            self.pneumatics_m2_state is None
            and self.pneumatics.evt_m2State.has_callback is False
        ):
            try:
                self.pneumatics_m2_state = (
                    await self.pneumatics.evt_m2State.aget(timeout=self.fast_timeout)
                ).state
            except asyncio.TimeoutError:
                self.log.warning("Could not get m2 valve state from ATPneumatics.")

            # set callback to monitor m2 valve state from now on...
            self.pneumatics.evt_m2State.callback = self.pneumatics_m2_state_callback

        # Instantiate the filter/disperser variables as they get
        # called below
        disperser_data = None
        filter_data = None
        if (
            self.atspectrograph_summary_state is None
            and self.atspectrograph.evt_summaryState.has_callback is False
        ):

            try:
                self.atspectrograph_summary_state = (
                    await self.atspectrograph.evt_summaryState.aget(
                        timeout=self.fast_timeout
                    )
                ).summaryState

            except asyncio.TimeoutError:
                self.log.warning("Could not get summary state from ATSpectrograph.")

            # Set callback for ATSpectrograph summary_state
            self.atspectrograph.evt_summaryState.callback = (
                self.atspectrograph_summary_state_callback
            )

            try:
                disperser_data = (
                    await self.atspectrograph.evt_reportedDisperserPosition.aget(
                        timeout=self.fast_timeout
                    )
                )
            except asyncio.TimeoutError:
                self.log.warning(
                    "Could not get reportedDisperserPosition from ATSpectrograph."
                )

            # Set callbacks for ATSpectrograph summary_state/filter/disperser
            # events
            self.atspectrograph.evt_reportedDisperserPosition.callback = (
                self.atspectrograph_disperser_monitor_callback
            )

            try:
                filter_data = await self.atspectrograph.evt_reportedFilterPosition.aget(
                    timeout=self.fast_timeout
                )
            except asyncio.TimeoutError:
                self.log.warning(
                    "Could not get reportedFilterPosition from ATSpectrograph."
                )

            # set callback for ATSpectrograph evt_reportedFilterPosition event
            self.atspectrograph.evt_reportedFilterPosition.callback = (
                self.atspectrograph_filter_monitor_callback
            )

        # Add focus offsets from spectrograph
        # These offsets are being applied for the first time. Because the
        # ATAOS was not monitoring the spectrograph when in the standby state
        # the offsets are made relative to nothing being in the beam

        if filter_data is not None and disperser_data is not None:
            self.focus_offset_per_category["filter"] = filter_data.focusOffset
            self.focus_offset_yet_to_be_applied += filter_data.focusOffset
            self.pointing_offsets_per_category["filter"] = np.array(
                filter_data.pointingOffsets
            )
            self.pointing_offsets_yet_to_be_applied += np.array(
                filter_data.pointingOffsets
            )
            self.current_atspectrograph_filter_name = filter_data.name
            self.current_atspectrograph_central_wavelength = (
                filter_data.centralWavelength
            )

            # Add a spectrograph correction loop so the offsets are applied
            self.atspectrograph_corrections_required += 1

            self.focus_offset_per_category["disperser"] = disperser_data.focusOffset
            self.focus_offset_yet_to_be_applied += disperser_data.focusOffset
            self.pointing_offsets_per_category["disperser"] = np.array(
                disperser_data.pointingOffsets
            )
            self.pointing_offsets_yet_to_be_applied += np.array(
                disperser_data.pointingOffsets
            )
            self.current_atspectrograph_disperser_name = disperser_data.name

        self.focus_offset_per_category["total"] = self.model.offset["z"]
        self.focus_offset_per_category["userApplied"] = 0.0

        await super().begin_start(data)
        self.log.debug("Completed begin_start")

    async def end_enable(self, data: type_hints.BaseMsgType) -> None:
        """End do_enable; called after state changes but before command
        acknowledged.

        It will add `self.correction_loop` to the event loop. The loop itself
        will be running but no corrections have been enabled, therefore it will
        just be looping over sleep statements

        Parameters
        ----------
        id_data : ``cmd_enable.DataType``
            Command data
        """

        # sets offsets to what they are in the init file, plus any
        # filter/disperser offsets set in begin_start
        await self.evt_correctionOffsets.set_write(
            **self.model.offset, force_output=True
        )
        await self.evt_focusOffsetSummary.set_write(
            **self.focus_offset_per_category, force_output=True
        )

        self.correction_loop_task = asyncio.ensure_future(self.correction_loop())

        self.log.debug(
            "At end of end_enable, correction loop now running but without "
            "the corrections enabled"
        )

    async def end_disable(self, data: type_hints.BaseMsgType) -> None:
        """End do_disable; called after state changes but before command
        acknowledged.

        Makes sure correction loop is cancelled appropriately.

        Parameters
        ----------
        data : ``cmd_disable.DataType``
            Command data

        """

        if not self.correction_loop_task.done():
            self.correction_loop_task.cancel()

        try:
            await self.correction_loop_task
        except asyncio.CancelledError:
            self.log.debug("Correction loop task cancelled.")
        except Exception as e:
            self.log.exception(
                "Exception while waiting for correction loop task to finish."
            )
            self.log.exception(e)

        correction_m1_before_disabling = self.corrections["m1"]
        correction_m2_before_disabling = self.corrections["m2"]

        disable = self.cmd_disableCorrection.DataType()
        disable.disableAll = True
        self.mark_corrections(disable, False)

        # Note that the mirror should only be lowered when
        # coming from the ENABLED state AND if corrections
        # were enabled

        if correction_m1_before_disabling or correction_m2_before_disabling:
            self.log.debug("Lower mirror to hardpoints")
            await self.lower_mirrors_to_hardpoints(
                m1=correction_m1_before_disabling, m2=correction_m2_before_disabling
            )
        else:
            self.log.debug("Both M1 and M2 corrections disabled. Not lowering mirrors.")

        await self.set_detailed_state(np.uint8(0))
        self.log.debug("At end of end_disable")

    async def do_applyCorrection(self, data: type_hints.BaseMsgType) -> None:
        """Apply a one-time correction based on model (LUTs) on all components
        either for the current position of the telescope (default) or the
        specified position.

        Corrections must be disabled to run this command.

        Since SAL does not allow definition of default parameters,
        azimuth = 0. and altitude = 0. is considered as "current telescope
        position". Note that, if altitude > 0 and azimuth=0, the correction is
        applied at the specified position.

        Angles wraps:
            azimuth: Absolute azimuth angle. Angle will be converted (wrapped)
             to the 0 - 360 range.
            elevation: (0., 90.] degrees (Model may still apply more
                restrictive limits)

        Parameters
        ----------
        data : ``cmd_applyCorrection.DataType``
            Command data

        Raises
        ------
        IOError
            If angles are outside bounds.
        AssertionError
            If one or more corrections are enabled.
        RuntimeError
            If telescope azimuth and/or elevation is not available
        """

        self.assert_enabled("applyCorrection")
        self.assert_corrections(enabled=False)

        await self.cmd_applyCorrection.ack_in_progress(
            data,
            timeout=self.apply_correction_timeout,
        )

        self.log.debug("Beginning do_applyCorrection")

        azimuth = data.azimuth % 360.0
        elevation = data.elevation

        if elevation == 0.0:
            if self.azimuth is None or self.elevation is None:
                raise RuntimeError(
                    "No information about telescope azimuth and/or "
                    "elevation available."
                )
            # Get telescope position stored by callback function.
            azimuth = self.azimuth
            elevation = self.elevation

        self.log.debug("Apply correction to Hexapod")
        await self.set_hexapod(azimuth, elevation)

        self.log.debug("Apply correction to M1")
        await self.set_pressure(
            "m1", azimuth, elevation, self.model.get_correction_m1(azimuth, elevation)
        )
        self.log.debug("Apply correction to M2")
        await self.set_pressure(
            "m2", azimuth, elevation, self.model.get_correction_m2(azimuth, elevation)
        )

        self.log.debug("Apply corrections from spectrograph")
        await self.set_atspectrograph_corrections(azimuth, elevation)

    async def do_applyFocusOffset(self, data: type_hints.BaseMsgType) -> None:
        """Applies a relative focus offset to the hexapod position
        correction model. This is a cumulative offset.
        The movement of the mirror is performed by the loop, not
        this function.

        The hexapod correction must be enabled to use this command.

        Parameters
        ----------
        data : ``cmd_applyFocusOffset.DataType``
            Command data

        Raises
        ------
        RuntimeError
            If hexapod correction is not enabled or a timeout occurred when
            applying the offsets

        Warning
        -------
        This method must not be called by any other method in the class,
        besides the regular CSC command-reply algorithm. If called
        internally, it may cause accounting errors in the offsets.
        """
        self.assert_enabled("applyFocusOffset")

        # Verify that correction loop is running
        if not self.corrections["hexapod"]:
            raise RuntimeError(
                "Hexapod correction is not enabled, no focus offset can be applied. "
                "Use enableCorrection to enable active correction, then apply the offsets "
                "using this command."
            )

        # Lock to apply the offset, which will only happen when the
        # correction loop is not running, therefore this will ensure
        # when the next correction occurs it will include the offset

        # Grab the asyncio lock so no offsets can be added while
        # the corrections are being applied

        async with self.correction_loop_lock:

            # Clear the last correction_loop_completed event
            self.correction_loop_completed_evt.clear()

            self.model.add_offset("z", data.offset)

            self.focus_offset_per_category["total"] += data.offset
            self.focus_offset_per_category["userApplied"] += data.offset

        # Lock now released, wait for correction to be applied.
        try:
            await asyncio.wait_for(
                self.correction_loop_completed_evt.wait(), self.cmd_timeout
            )
        except asyncio.TimeoutError:
            raise RuntimeError(
                "Timed out when waiting for correction loop "
                "to apply the offsets during the "
                "applyFocusOffset command."
            )

        # Publish new offset values
        await self.evt_correctionOffsets.set_write(
            **self.model.offset, force_output=True
        )
        self.evt_focusOffsetSummary.set_write(
            **self.focus_offset_per_category, force_output=True
        )

    async def do_setCorrectionModelOffsets(self, data: type_hints.BaseMsgType) -> None:
        """Sets offset to selected model axis correction. These are *NOT* a
        cumulative offsets and will reset whatever values are currently
        entered. Use with caution.

        Any values determined from the spectrograph configuration will also
        be reset, except pointing offsets.

        The hexapod correction must be enabled to use this command.

        Parameters
        ----------
        data : ``cmd_setCorrectionModelOffsets.DataType``
            Command data

        Raises
        ------
        RuntimeError
            If hexapod correction is not enabled.
        """
        self.assert_enabled("setCorrectionModelOffsets")

        # Verify that correction loop is running
        if not self.corrections["hexapod"]:
            raise RuntimeError(
                "Hexapod correction is not enabled, no focus offset can be applied. "
                "Use enableCorrection to enable active correction, then apply the offsets "
                "using this command."
            )

        # Grab the asyncio lock so no offsets can be added while
        # the corrections are being applied
        async with self.correction_loop_lock:

            self.model.set_offset(data.axis, data.offset)

            # Wait for correction to be applied.
            try:
                await asyncio.wait_for(
                    self.correction_loop_completed_evt.wait(), self.cmd_timeout
                )
            except asyncio.TimeoutError:
                raise RuntimeError(
                    "Timed out when waiting for correction loop "
                    "to apply the offsets during the "
                    "setCorrectionModelOffsets command"
                )

        await self.evt_correctionOffsets.set_write(
            **self.model.offset, force_output=True
        )
        # reset all offsets to zero
        if data.axis == "z":
            self.focus_offset_per_category["total"] = data.offset
            self.focus_offset_per_category["userApplied"] = data.offset
            self.focus_offset_per_category["filter"] = 0.0
            self.focus_offset_per_category["disperser"] = 0.0
            self.focus_offset_per_category["wavelength"] = 0.0
            await self.evt_focusOffsetSummary.set_write(
                **self.focus_offset_per_category, force_output=True
            )

    async def do_offset(self, data: type_hints.BaseMsgType) -> None:
        """Apply relative offsets to any axis of the model (m1, m2, and
        hexapod x,y,z,u,v).

        Offsets are cumulative, e.g. if the command is sent twice, with the
        same values you will get double the amount of offset.

        Offsets can only be applied when the correction loop is enabled.
        Enable corrections using the enableCorrection command.

        Parameters
        ----------
        data : ``cmd_offset.DataType``
            Command data

        Raises
        ------
        RuntimeError
            If appropriate corrections are note enabled or
            application of offsets by the correction loop times out

        Warning
        -------
        This method must not be called by any other method in the class,
        besides the regular CSC command-reply algorithm. If called internally,
        it may cause accounting errors in the offsets.
        """
        self.assert_enabled("offset")

        # Verify that the appropriate correction loop is running
        # and not all offsets are zero (1e-12 to prevent possible
        # floating point errors
        if (
            any([abs(getattr(data, axis)) > 1e-12 for axis in "xyzuv"])
            and not self.corrections["hexapod"]
        ):
            raise RuntimeError(
                "Hexapod correction is not enabled. Offsets cannot be applied. "
                "See the enableCorrection to close the correction loop."
            )
        elif (abs(data.m1) > 1e-12) and not self.corrections["m1"]:
            raise RuntimeError(
                "M1 correction is not enabled. Offset cannot be applied. "
                "See the enableCorrection to close the correction loop."
            )
        elif (abs(data.m2) > 1e-12) and not self.corrections["m2"]:
            raise RuntimeError(
                "M2 correction is not enabled. Offset cannot be applied. "
                "See the enableCorrection to close the correction loop."
            )

        # Acquire the lock before applying correction
        async with self.correction_loop_lock:
            # Add offsets to the model, then release the lock
            for axis in self.model.offset:
                self.model.add_offset(axis, getattr(data, axis))

        # Wait for offset to be applied in the loop, first clear past event
        self.correction_loop_completed_evt.clear()

        try:
            await asyncio.wait_for(
                self.correction_loop_completed_evt.wait(), self.cmd_timeout
            )
        except asyncio.TimeoutError:
            raise RuntimeError(
                "Timed out when waiting for correction loop "
                "to apply offsets during offset command"
            )

        self.log.debug("sending evt_correctionOffsets")
        await self.evt_correctionOffsets.set_write(
            **self.model.offset, force_output=True
        )

        # Only sending the focusOffsetSummary event if a focus affecting
        # (z-axis) offset is applied (not equal to zero)
        if abs(getattr(data, "z")) > 1e-12:
            self.focus_offset_per_category["total"] += getattr(data, "z")
            self.focus_offset_per_category["userApplied"] += getattr(data, "z")
            await self.evt_focusOffsetSummary.set_write(
                **self.focus_offset_per_category, force_output=True
            )

    async def do_resetOffset(self, data: type_hints.BaseMsgType) -> None:
        """Reset userApplied provided offsets on a specific axis or all.
        Grating/Filter focus and pointing offsets will remain.

        Offsets can only be reset if the appropriate correction loop is closed.

        Parameters
        ----------
        data : ``cmd_resetOffset.DataType``
            Command data

        Raises
        ------
        RuntimeError
            If appropriate corrections are not enabled, illegal axis
            value is specified, or corrections timeout when being
            applied in the correction loop.
        """
        self.assert_enabled("resetOffset")

        # Grab the asyncio lock so no offsets can be added while
        # the corrections are being applied
        # Ideally this would be applied before the if statements below
        # but that would make the code clunky. Seeing as this is rarely used
        # and the expressions are fast we'll apply the lock before the check.

        async with self.correction_loop_lock:

            if len(data.axis) == 0 or data.axis == "all":
                # Verify all loops are closed
                if not (
                    self.corrections["hexapod"]
                    and self.corrections["m1"]
                    and self.corrections["m2"]
                ):
                    raise RuntimeError(
                        "Not all corrections are enabled. "
                        "Offsets cannot be reset as a group. "
                        "See the enableCorrection to close the correction loop "
                        "or reset individual axes."
                    )
                self.log.debug("Resetting all offsets")
                self.model.reset_offset()
            elif data.axis in self.model.offset:
                # Verify appropriate corrections are enabled
                if data.axis == "m1" and not self.corrections["m1"]:
                    raise RuntimeError(
                        "m1 correction must be enabled to reset the offset."
                    )
                elif data.axis == "m2" and not self.corrections["m2"]:
                    raise RuntimeError(
                        "m2 correction must be enabled to reset the offset."
                    )
                elif data.axis in "xyzuv" and not self.corrections["hexapod"]:
                    raise RuntimeError(
                        "hexapod correction must be enabled to reset the offset."
                    )

                # Apply the offset
                if self.corrections["atspectrograph"] and data.axis == "z":
                    # If reseting focus (axis = z) and atspectrograph
                    # correction is on then we must add back in the filter,
                    # grating and wavelength offsets best to do this
                    # immediately and not later in a loop as it'll be confusing
                    # to see large offsets pop up when the loop comes on rather
                    # then when the reset command is set.
                    _offsetToApply = (
                        self.focus_offset_per_category["filter"]
                        + self.focus_offset_per_category["disperser"]
                        + self.focus_offset_per_category["wavelength"]
                    )
                    self.model.set_offset("z", _offsetToApply)
                else:
                    self.model.set_offset(data.axis, 0.0)

            else:
                raise RuntimeError(
                    f"Axis {data.axis} invalid. Must be one of "
                    "m1, m2, x, y, z, u, v, all or an empty string with length zero."
                )

            # Do not reset the filter/grating offsets, but reset the others
            self.focus_offset_per_category["total"] = self.model.offset["z"]
            self.focus_offset_per_category["userApplied"] = 0.0

            # Now want to await for offsets to be reset via the loop,
            # but first clear past event, then release the lock
            self.correction_loop_completed_evt.clear()

        # Wait for offsets to be applied
        try:
            await asyncio.wait_for(
                self.correction_loop_completed_evt.wait(), self.cmd_timeout
            )
        except asyncio.TimeoutError:
            raise RuntimeError(
                "Timed out when waiting for correction loop "
                "to apply offsets during resetOffset command"
            )

        # Publish events
        await self.evt_focusOffsetSummary.set_write(
            **self.focus_offset_per_category, force_output=True
        )
        await self.evt_correctionOffsets.set_write(
            **self.model.offset, force_output=True
        )

    async def do_enableCorrection(self, data: type_hints.BaseMsgType) -> None:
        """Enable corrections on specified axes.

        This method only works for enabling corrections, it will not disable
        any of the corrections (including `move_while_exposing`).

        Setting components to False in this command will not cause any affect,
        e.g. if m1 correction is enabled and `enable_correction` receives
        `id_data.m1=False`, the correction will remain enabled.
         To disable a correction (or `move_while_exposing`) use
        `do_disableCorrection()`.

        Parameters
        ----------
        data : ``cmd_enableCorrection.DataType``
            Command data.

        Raises
        ------
        Exception
            If m1 or m2 valve fails to open
        RuntimeError
            If correction loop fails to complete just before returning
            from this method.
        """
        self.assert_enabled("enableCorrection")
        self.assert_any_corrections(data)

        await self.cmd_enableCorrection.ack_in_progress(
            data,
            timeout=self.enable_correction_timeout,
        )

        self.log.debug("Enabling corrections.")

        try:
            if data.m1 or data.enableAll:
                self.log.debug("Preparing M1.")
                await self.check_atpneumatic()
                try:
                    if self.pneumatics_m1_state != ATPneumatics.AirValveState.OPENED:
                        self.log.debug("Opening m1 air valve.")
                        await self.pneumatics.cmd_m1OpenAirValve.start(
                            timeout=self.cmd_timeout
                        )
                except AckError as ack_error:
                    if ack_error.ackcmd.ack == SalRetCode.CMD_NOPERM:
                        self.log.exception("M1 valve is already opened.")
                    else:
                        RuntimeError("Failed to open m1 air valve.")

                # FIXME DM-36212:
                # ATPneumatics is not ready to set pressure just after
                # the valve is opened and there is currently no event to
                # indicate readiness. Wait 3 heartbeat intervals after command
                # finishes. A single heartbeat was not sufficient.
                # Once this is fixed the sleep can be removed.
                await asyncio.sleep(3 * self.heartbeat_interval)

                # Set pressure to zero.
                await self.pneumatics.cmd_m1SetPressure.set_start(
                    pressure=0.0, timeout=self.cmd_timeout
                )

                self.log.debug("M1 ready.")
        except Exception as e:
            self.log.exception("Failed to open m1 air valve.")
            raise e

        try:
            if data.m2 or data.enableAll:
                self.log.debug("Preparing M2.")
                await self.check_atpneumatic()
                try:
                    if self.pneumatics_m2_state != ATPneumatics.AirValveState.OPENED:
                        await self.pneumatics.cmd_m2OpenAirValve.start(
                            timeout=self.cmd_timeout
                        )
                except AckError as ack_error:
                    if ack_error.ackcmd.ack == SalRetCode.CMD_NOPERM:
                        self.log.exception("M2 valve is already opened.")
                    else:
                        raise ack_error

                # set the pressure to zero before opening the valve so no
                # large pressure spikes occur can't set it before opening
                # valve or it'll be rejected
                await self.pneumatics.cmd_m2SetPressure.set_start(
                    pressure=0.0, timeout=self.cmd_timeout
                )
                self.log.debug("M2 ready.")
        except Exception as e:
            self.log.exception("Failed to open m2 air valve.")
            raise e

        # make sure the spectrograph is still online and enabled
        if data.atspectrograph or data.enableAll:
            await self.check_atspectrograph()

        self.mark_corrections(data, True)

        # Perform one correction loop before handing back
        # Wait for correction to be applied - should be fast unless large
        # hexap position changes are required.
        self.correction_loop_completed_evt.clear()

        try:
            await asyncio.wait_for(
                self.correction_loop_completed_evt.wait(), self.cmd_timeout
            )
        except asyncio.TimeoutError:
            raise RuntimeError(
                "Timed out when waiting for correction loop "
                "to complete during enableCorrection"
            )

        await self.publish_enable_corrections()

    async def do_disableCorrection(self, data: type_hints.BaseMsgType) -> None:
        """Disable corrections on specified axis.

        This is the mirror method of `enable_correction`, and will only
        disable features (including `move_while_exposing`).

        Parameters
        ----------
        data : ``cmd_disableCorrection.DataType``
            Command data
        """
        self.assert_enabled("disableCorrection")
        self.assert_any_corrections(data)

        # give control back to event loop such that other operations can
        # be performed while this function is running
        await asyncio.sleep(0.0)

        self.mark_corrections(data, False)
        await asyncio.sleep(0.0)  # give control back to event loop

        # Lower mirrors if appropriate
        if data == "disableAll":
            await self.lower_mirrors_to_hardpoints(m1=True, m2=True)
        elif data == "m1":
            await self.lower_mirrors_to_hardpoints(m1=True, m2=False)
        elif data == "m2":
            await self.lower_mirrors_to_hardpoints(m1=False, m2=True)

        await self.publish_enable_corrections()

    async def do_setWavelength(self, data: type_hints.BaseMsgType) -> None:
        """Set wavelength to optimize focus when being used with the
        LATISS spectrograph. This must only be used
        when the spectrograph is being used (glass optics are in the beam).
        This function applies the offset to the model, whereas the loop
        will apply the actual movement of the hexapod position.

        This functionality requires verification on-sky. It is possible
        that a single relationship for all instrument setups will not be
        sufficient due to variation in glass thickness/dispersions.

        Note that the offset due to wavelength will be zero'd out with
        every filter change. The focus offset value for each filter is
        for the specified central wavelength of that filter in the
        configuration file (ts_config_attcs).

        Parameters
        ----------
        id_data : ``cmd_setWavelength.DataType``
            Command data

        Raises
        ------
        RuntimeError
            If hexapod corrections are not enabled, the
            beam contains a filter and/or disperser, or the
            corrections fail to get applied before timing out.

        Warning
        -------
        This method must not be called by any other method in the class,
        besides the regular CSC command-reply algorithm. If called
        internally, it may cause accounting errors in the offsets.
        """
        self.assert_enabled("setWavelength")

        # Verify the atspectrograph and hexapod corrections are enabled
        if not (self.corrections["hexapod"] and self.corrections["atspectrograph"]):
            raise RuntimeError(
                "hexapod at atspectrograph corrections must "
                "be enabled (via enableCorrection) to apply "
                "wavelength offsets."
            )

        # Check that there is a filter or disperser in the beam (meaning a
        # non-zero focus offset for a filter is in place). If the beam is
        # empty then this command should not be applied as the telescope
        # is achromatic.

        if (
            abs(self.focus_offset_per_category["filter"])
            + abs(self.focus_offset_per_category["disperser"])
            == 0.0
        ):
            raise RuntimeError(
                "Focus offsets associated with the filter/disperser "
                f" ({self.current_atspectrograph_filter_name}/"
                f"{self.current_atspectrograph_disperser_name})"
                f" are zero. No wavelength offsets permitted unless"
                "an optic is in the beam."
            )

        # Grab the asyncio lock when applying the correction

        async with self.correction_loop_lock:
            self.log.debug(
                "Compensating for focus difference due to change in wavelength."
            )

            # must subtract whatever chromatic offset might already be in place
            _chromatic_offset = (
                self.model.get_correction_chromatic(data.wavelength)
                - self.focus_offset_per_category["wavelength"]
            )

            self.model.add_offset("z", _chromatic_offset)

            self.focus_offset_per_category["total"] += _chromatic_offset
            self.focus_offset_per_category["wavelength"] = _chromatic_offset

            # add a correction cycle to make the wavelength change
            self.atspectrograph_corrections_required += 1

        # Perform one correction loop before handing back
        # Wait for correction to be applied - should be fast unless large
        # hexap position changes are required.
        self.correction_loop_completed_evt.clear()

        try:
            await asyncio.wait_for(
                self.correction_loop_completed_evt.wait(), self.cmd_timeout
            )
        except asyncio.TimeoutError:
            raise RuntimeError(
                "Timed out when waiting for correction loop "
                "to complete when performing setWavelength "
                "offset. Corrections may not have been"
                " applied."
            )

        # Publish event summarizing the offsets
        await self.evt_focusOffsetSummary.set_write(
            **self.focus_offset_per_category, force_output=True
        )
        await self.evt_correctionOffsets.set_write(
            **self.model.offset, force_output=True
        )

    async def correction_loop(self) -> None:
        """Coroutine to send corrections to m1, m2, hexapod and focus at
        the heartbeat frequency."""

        while self.summary_state == State.ENABLED:
            # Create a sleep task that will help keep the loop frequency
            # near the target by running concurrently with the loop
            sleep_task: typing.Awaitable = asyncio.create_task(
                asyncio.sleep(self.correction_loop_time)
            )

            # Grab the asyncio lock so no offsets can be added while
            # the corrections are being applied
            async with self.correction_loop_lock:

                try:
                    corrections_to_apply = []
                    if self.azimuth is not None and self.elevation is not None:
                        elevation = self.elevation
                        azimuth = self.azimuth

                        if (
                            self.target_elevation is not None
                            and self.target_elevation < self.elevation
                        ):
                            # Telescope in going down, need to go ahead and
                            # decrease pressure accordingly
                            elevation = (self.target_elevation + self.elevation) / 2.0
                            self.log.debug(
                                "Telescope going down, getting ahead on correction."
                                f"el: {self.elevation}, target_el: {self.target_elevation}, "
                                f"corr_el: {elevation}"
                            )

                        for correction in self.corrections:
                            # corrections_routines only has m1, m2 and hexapod,
                            # NOT FOCUS
                            if (
                                self.corrections[correction]
                                and correction in self.corrections_routines
                            ):
                                self.log.debug(f"Adding {correction} correction.")
                                corrections_to_apply.append(
                                    self.corrections_routines[correction](
                                        azimuth, elevation
                                    )
                                )

                        # Check to see if any model offsets need applying due
                        # to changes in atspectrograph filter/disperser
                        # changes, then apply if needed
                        self.log.debug(
                            f"atspectrograph: {self.corrections['atspectrograph']}, "
                            f"focus_offset_yet_to_be_applied: {self.focus_offset_yet_to_be_applied}, "
                            f"pointing_offsets_yet_to_be_applied: {self.pointing_offsets_yet_to_be_applied}"
                        )

                    else:
                        # This should not raise an exception as it would
                        # require the ATMCS is online for the ATAOS to
                        # be online. Having the loop running (but just
                        # sleeping) is acceptable and allows cycling of
                        # components.
                        self.log.debug(
                            "No information available about telescope azimuth and/or "
                            "elevation, m1, m2, hexapod corrections cannot occur "
                        )

                    # FIXME: DM-28681
                    # Run corrections in series because CSCs are not
                    # supporting concurrent operations yet 2019/June/4

                    for corr in corrections_to_apply:
                        await corr

                    # run corrections concurrently (and/or wait for the
                    # heartbeat interval)
                    # if len(corrections_to_apply) > 0:
                    #     await asyncio.gather(*corrections_to_apply)

                except asyncio.CancelledError:
                    self.log.debug("Correction loop cancelled.")
                    break
                except Exception:
                    self.log.exception(
                        "Error in correction loop. Going to FAULT state."
                    )
                    await self.fault(
                        code=CORRECTION_LOOP_DIED,
                        report="Correction loop died.",
                        traceback=traceback.format_exc(),
                    )
                    break

            # Lock releases when outside the "with" statement
            # Set an asyncio.Event (not a DDS Event) saying the loop has
            # finished and lock is released
            self.correction_loop_completed_evt.set()
            # await any remaining time (up to the loop time) before
            # starting the next iteration
            await sleep_task

    async def lower_mirrors_to_hardpoints(
        self, m1: bool = True, m2: bool = True
    ) -> None:
        """Lower mirrors on to hardpoints by setting pneumatic pressures
        to zero. This is to be called whenever safety of the mirror, or
        lifting of the mirror from the hardpoints (completely) may be a
        concern.

        This method is expected to be called when disabling the CSC,
        transitioning to fault state, and disabling corrections.

        Parameters
        ----------
        m1 : `boolean`
            Lower primary (m1) mirror

        m2 : `boolean`
            Lower secondary (m2) mirror
        """

        # Try statements required to handle case when ATPneumatics is
        # disabled.
        try:
            if m1:
                # Setting m1 pressure to zero and close valve
                self.pneumatics.cmd_m1SetPressure.set(pressure=0.0)
                await self.pneumatics.cmd_m1SetPressure.start(timeout=self.cmd_timeout)
                await self.pneumatics.cmd_m1CloseAirValve.start(
                    timeout=self.cmd_timeout
                )
                self.log.info("M1 mirror lowered onto hardpoints")
        except Exception:
            self.log.exception(
                "Failed to set m1 presssure to zero and/or close m1 air valve."
            )

        try:
            if m2:
                # Setting m2 pressure to zero and close valve
                self.pneumatics.cmd_m2SetPressure.set(pressure=0.0)
                await self.pneumatics.cmd_m2SetPressure.start(timeout=self.cmd_timeout)
                await self.pneumatics.cmd_m2CloseAirValve.start(
                    timeout=self.cmd_timeout
                )
                self.log.info("M2 mirror lowered onto hardpoints")
        except Exception:
            self.log.exception(
                "Failed to set m1 presssure to zero and/or close m2 air valve."
            )

    def assert_any_corrections(self, data: type_hints.BaseMsgType) -> None:
        """Check that at least one attribute of
        disableCorrection or enableCorrection are set to True.

        Parameters
        ----------
        data : ``cmd_disableCorrection.DataType`` or
               ``cmd_enableCorrection.DataType``
            Command data

        Raises
        ------
        AssertionError
            If one of more attribute of the topic is set to True.
        """

        assert any(
            [getattr(data, corr, False) for corr in self.valid_corrections]
        ), "At least one correction must be set."

    def assert_corrections(self, enabled: bool) -> None:
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
            enabled_keys = [
                key for key, is_enabled in self.corrections.items() if is_enabled
            ]
            assert not any(self.corrections.values()), "Corrections %s enabled: %s." % (
                enabled_keys,
                self.corrections.items(),
            )

    def can_move(self) -> bool:
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

    def mark_corrections(self, data: type_hints.BaseMsgType, flag: bool) -> None:
        """Utility method to switch corrections on/off.

        Parameters
        ----------
        data : ``cmd_disableCorrection.DataType`` or
               ``cmd_enableCorrection.DataType``
            Command data
        flag : bool
            Value to set the corrections to
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

    async def publish_enable_corrections(self) -> None:
        """Utility function to publish enable corrections."""
        kwargs = dict((key, value) for key, value in self.corrections.items())
        await self.evt_correctionEnabled.set_write(
            moveWhileExposing=self.move_while_exposing, **kwargs
        )

    def shutter_monitor_callback(self, data: type_hints.BaseMsgType) -> None:
        """A callback function to monitor the camera shutter.

        Parameters
        ----------
        data : `ATCamera_logevent_shutterDetailedState`
            Sample of shutterDetailedState event from ATCamera
        """
        self.camera_exposing = data.substate != ShutterState.CLOSED

    def atspectrograph_filter_monitor_callback(
        self, data: type_hints.BaseMsgType
    ) -> None:
        """A callback function to monitor the atspectrograph filter/grating
        changes.

        Parameters
        ----------
        data : `ATSpectrograph_logevent_reportedFilterPosition`
            Sample of reportedFilterPosition event from ATSpectrograph
        """
        # Do not want to apply any corrections here.
        # Method is to track the corrections to be applied, even if the
        # correction is disabled

        self.log.info(
            "Caught ATSpectrograph filter change, calculating correction offsets"
        )
        # Add a correction cycle to accommodate the change
        self.atspectrograph_corrections_required += 1
        # Relative offset are to be applied to the model
        # so therefore we need to subtract offset already in place for the
        # previous filter, including the wavelength offset setting
        _offset_to_apply = (
            data.focusOffset
            - self.focus_offset_per_category["filter"]
            - self.focus_offset_per_category["wavelength"]
        )
        _pointing_offsets_to_apply = (
            np.array(data.pointingOffsets)
            - self.pointing_offsets_per_category["filter"]
        )

        self.log.debug(
            "atspectrograph changed filters "
            f"from {self.current_atspectrograph_filter_name} to {data.name}"
        )
        self.log.debug(
            f"Calculated a hexapod-z model offset of {_offset_to_apply} "
            "based on focus differences between filters"
        )
        self.log.debug(
            f"Calculated a pointing offset of {_pointing_offsets_to_apply} "
            "based on focus differences between filter data"
        )

        self.current_atspectrograph_filter_name = data.name
        self.current_atspectrograph_central_wavelength = data.centralWavelength
        # Apply the offsets to the focusOffsetSummary event
        self.focus_offset_per_category["total"] += _offset_to_apply
        self.focus_offset_per_category["filter"] = data.focusOffset
        # Always default focus to filter central wavelength
        self.focus_offset_per_category["wavelength"] = 0.0
        self.focus_offset_yet_to_be_applied += _offset_to_apply

        self.pointing_offsets_per_category["filter"] = np.array(data.pointingOffsets)
        self.pointing_offsets_yet_to_be_applied += _pointing_offsets_to_apply

    def atspectrograph_disperser_monitor_callback(
        self, data: type_hints.BaseMsgType
    ) -> None:
        """A callback function to monitor the atspectrograph filter/grating
        changes.

        Parameters
        ----------
        data : `ATSpectrograph_logevent_reportedDisperserPosition`
            Sampe of reportedDisperserPosition event from ATSpectrograph
        """
        self.log.info(
            "Caught ATSpectrograph disperser change, calculating correction offsets"
        )
        # Add a correction loop cycle
        self.atspectrograph_corrections_required += 1
        _offset_to_apply = (
            data.focusOffset - self.focus_offset_per_category["disperser"]
        )
        _pointing_offsets_to_apply = (
            data.pointingOffsets - self.pointing_offsets_per_category["disperser"]
        )

        self.log.debug(
            "atspectrograph changed dispersers "
            f"from {self.current_atspectrograph_disperser_name} to {data.name}"
        )
        self.log.debug(
            f"Calculated a hexapod-z model offset of {_offset_to_apply} "
            "based on focus differences between dispersers"
        )
        self.log.debug(
            f"Calculated a pointing offset of {_pointing_offsets_to_apply} "
            "based on focus differences between filter data"
        )

        self.current_atspectrograph_disperser_name = data.name

        # Apply the offsets to the focusOffsetSummary event
        self.focus_offset_per_category["total"] += _offset_to_apply
        self.focus_offset_per_category["disperser"] = data.focusOffset
        self.focus_offset_yet_to_be_applied += _offset_to_apply

        self.pointing_offsets_per_category["disperser"] = np.array(data.pointingOffsets)
        self.pointing_offsets_yet_to_be_applied += _pointing_offsets_to_apply

    def hexapod_monitor_callback(self, data: type_hints.BaseMsgType) -> None:
        """A callback function to monitor position updates on the hexapod.

        Parameters
        ----------
        data : `ATHexapod_logevent_positionUpdate`
            Sample positionUpdate event from ATHexapod
        """
        self.current_positions["x"] = data.positionX
        self.current_positions["y"] = data.positionY
        self.current_positions["z"] = data.positionZ
        self.current_positions["u"] = data.positionU
        self.current_positions["v"] = data.positionV

    def m1_pressure_monitor_callback(self, data: type_hints.BaseMsgType) -> None:
        """Callback function to monitor M1 pressure

        Parameters
        ----------
        data : `ATPneumatics_m1AirPressure`
            Sample m1AirPressure telemtry from  ATPneumatics
        """
        self.current_positions["m1"] = data.pressure

    def m2_pressure_monitor_callback(self, data: type_hints.BaseMsgType) -> None:
        """Callback function to monitor M2 pressure

        Parameters
        ----------
        data : `ATPneumatics_m2AirPressure`
            Sample m2AirPressure telemtry from  ATPneumatics
        """
        self.current_positions["m2"] = data.pressure

    async def set_pressure_m1(self, azimuth: float, elevation: float) -> None:
        """Set pressure on m1.

        Parameters
        ----------
        azimuth : float
        elevation : float
        """
        pressure = self.model.get_correction_m1(azimuth, elevation)

        await self.set_pressure(
            "m1",
            azimuth,
            elevation,
            pressure,
            self.current_positions["m1"],
            self.correction_tolerance["m1"],
        )

    async def set_pressure_m2(self, azimuth: float, elevation: float) -> None:
        """Set pressure on m2.

        Parameters
        ----------
        azimuth : float
        elevation : float
        """
        pressure = self.model.get_correction_m2(azimuth, elevation)

        await self.set_pressure(
            "m2",
            azimuth,
            elevation,
            pressure,
            self.current_positions["m2"],
            self.correction_tolerance["m2"],
        )

    async def set_pressure(
        self,
        mirror: str,
        azimuth: float,
        elevation: float,
        pressure: float,
        current: typing.Optional[float] = None,
        tolerance: typing.Optional[float] = None,
    ) -> None:
        """Set pressure on specified mirror.

        If ``current`` and ``tolerance`` are set, pressure will only be set if
        the difference between  ``pressure`` and ``current`` is larger than
        ``tolerance``.

        Parameters
        ----------
        mirror : str
            Either m1 or m2
        azimuth : float
            Azimuth position (deg)
        elevation : float
            Elevation position (deg)
        pressure : float
            Pressure (Pa)
        current : float
            Current pressure (Pa)
        tolerance : float
            Tolerance in pressure value (Pa)
        """
        status_bit = getattr(DetailedState, f"{mirror}".upper())

        # Check that pressure is not being applied yet
        if self.detailed_state & status_bit != 0:
            self.log.warning("%s pressure correction running... skipping...", mirror)
            return
        elif (
            current is not None
            and tolerance is not None
            and tolerance > 0.0
            and np.abs(current - pressure) < tolerance
        ):
            self.log.debug(
                f"Set value ({pressure}) and current value ({current}) "
                f"inside tolerance ({tolerance}). Ignoring."
            )
            return
        else:
            # publish new detailed state
            await self.set_detailed_state(self.detailed_state ^ status_bit)

            cmd_attr = getattr(self.pneumatics, f"cmd_{mirror}SetPressure")
            evt_start_attr = getattr(self, f"evt_{mirror}CorrectionStarted")
            evt_end_attr = getattr(self, f"evt_{mirror}CorrectionCompleted")

            await evt_start_attr.set_write(
                azimuth=azimuth, elevation=elevation, pressure=pressure
            )
            try:
                await cmd_attr.set_start(pressure=pressure, timeout=self.cmd_timeout)
            except Exception as e:
                self.log.exception(
                    f"Failed to set pressure for {mirror} @ "
                    f"AzEl: {azimuth}/{elevation}"
                )
                raise e
            finally:
                await evt_end_attr.set_write(
                    azimuth=azimuth, elevation=elevation, pressure=pressure
                )
                # correction completed... flip bit on detailedState
                await self.set_detailed_state(self.detailed_state ^ status_bit)

    async def set_hexapod(
        self, azimuth: float, elevation: float, axis: str = "xyzuvw"
    ) -> None:
        """Utility to calculate desired hexapod position based on models,
         then apply the movements.

        Parameters
        ----------
        azimuth : float
            Azimuth (deg)
        elevation : float
            Elevation (deg)
        axis : string
            Which axis to set position. Any subset of the string "xyzuvw".
        """

        if self.can_move():
            # publish new detailed state

            self.log.debug(f"Moving hexapod axis {axis}")

            status_bit = DetailedState.HEXAPOD

            evt_start_attr = getattr(self, "evt_hexapodCorrectionStarted")
            evt_end_attr = getattr(self, "evt_hexapodCorrectionCompleted")

            if axis == "z":
                status_bit = DetailedState.FOCUS
                evt_start_attr = getattr(self, "evt_focusCorrectionStarted")
                evt_end_attr = getattr(self, "evt_focusCorrectionCompleted")

            await self.set_detailed_state(np.uint8(self.detailed_state ^ status_bit))

            hexapod = dict(
                zip(
                    [f"hexapod_{ax}" for ax in "xyzuvw"],
                    self.model.get_correction_hexapod(azimuth, elevation),
                )
            )

            hexapod_mov = dict(
                zip("xyzuvw", self.model.get_correction_hexapod(azimuth, elevation))
            )

            apply_correction = False

            for axis in hexapod_mov:
                if axis == "w":
                    continue
                current = self.current_positions[axis]
                tolerance = self.correction_tolerance[axis]
                set_value = hexapod_mov[axis]

                if (
                    current is not None
                    and tolerance is not None
                    and tolerance > 0.0
                    and np.abs(current - set_value) < tolerance
                ):
                    self.log.debug(
                        f"Set value ({set_value}) and current value ({current}) "
                        f"inside tolerance ({tolerance}) for axis {axis}. Ignoring."
                    )
                    continue
                else:
                    self.log.debug(
                        f"Set value for axis {axis} exceeds minimal threshold "
                        f"required to apply correction. Applying correction."
                    )
                    apply_correction = True
                    break

            if not apply_correction:
                self.log.debug("All hexapod corrections inside tolerance. Skipping...")
                # Flip bit back before returning
                await self.set_detailed_state(
                    np.uint8(self.detailed_state ^ status_bit)
                )
                return

            evt_start_attr.set(elevation=elevation, azimuth=azimuth, **hexapod)

            evt_end_attr.set(elevation=elevation, azimuth=azimuth, **hexapod)

            await evt_start_attr.write()

            try:
                await self.hexapod.cmd_moveToPosition.set_start(
                    **hexapod_mov, timeout=self.cmd_timeout
                )
            except Exception as e:
                self.log.warning(
                    f"Failed to set hexapod position @ " f"AzEl: {azimuth}/{elevation}"
                )
                self.log.exception(e)
                raise e
            finally:
                await evt_end_attr.write()
                # correction completed... flip bit on detailedState
                # back to IDLE
                await self.set_detailed_state(
                    np.uint8(self.detailed_state ^ status_bit)
                )

    async def set_atspectrograph_corrections(
        self, azimuth: float, elevation: float
    ) -> None:
        """Utility to apply hexapod and/or pointing corrections based on
         the ATSpectrograph configuration. This gets called as part of
         the regular control loop.

        Parameters
        ----------
        azimuth : float
            Azimuth (deg)
        elevation : float
            Elevation (deg)

        Warning
        -------
        Can only be applied if hexapod correction is also applied.
        """
        self.log.debug(
            f"At the beginning of set_atspectrograph_corrections with "
            f"{self.atspectrograph_corrections_required} correction "
            f"loops required. "
        )

        # The following warnings should never occur but are here to help
        # troubleshoot the issue should it arise.
        if (
            self.atspectrograph_corrections_required == 0
            and abs(self.focus_offset_yet_to_be_applied) >= 1e-12
        ):
            self.log.warning(
                f"No atspectrograph corrections loaded but focus_offsets "
                f"of {self.focus_offset_yet_to_be_applied} are "
                f"non-zero and above the numerical noise floor"
            )
        if (
            self.atspectrograph_corrections_required == 0
            and np.max(np.abs(self.pointing_offsets_yet_to_be_applied)) >= 1e-12
        ):
            self.log.warning(
                "No atspectrograph corrections loaded but pointing_offsets of"
                f" {self.pointing_offsets_yet_to_be_applied} are non-zero and"
                f" above the numerical noise floor"
            )

        # if self.can_move() and
        # (self.atspectrograph_corrections_required != 0):
        # Ideally, only the check against can_move and
        # atspectrograph_corrections_required should be necessary, however
        # leaving the others in and will search the logs for the above
        # warning messages just in case a bug exists.
        if self.can_move() and (
            np.max(np.abs(self.pointing_offsets_yet_to_be_applied)) >= 1e-12
            or abs(self.focus_offset_yet_to_be_applied) >= 1e-12
            or self.atspectrograph_corrections_required != 0
        ):
            self.log.debug(
                "Applying Spectrograph Corrections due to filter/grating/wavelength change"
            )

            status_bit = DetailedState.ATSPECTROGRAPH
            _offset_value = 0.0
            # tolerance in arcseconds. Value is arbitrary but here for
            # rounding issues
            _pointing_tolerance = 0.01
            _pointing_offsets = np.zeros((2))

            # Start with Focus offset

            # check that offset is not just numerical noise. In our case
            # anything smaller than the correction tolerance won't be
            # applied to the hexapod anyways, so any offset 10x smaller
            # than that is surely noise.

            self.log.debug(
                f"Applying focus offset of {self.focus_offset_yet_to_be_applied} "
                "in correction loop due to filter "
                "and/or disperser changes."
            )
            # add the offset, then reset the value
            # using subtraction here to avoid a possible race condition
            # note that self.focus_offset_yet_to_be_applied is a value
            # not an object so no deepcopy is required
            if abs(self.focus_offset_yet_to_be_applied) > abs(
                self.correction_tolerance["z"] / 10.0
            ):
                _offset_value = self.focus_offset_yet_to_be_applied
                self.model.add_offset("z", _offset_value)
                self.focus_offset_yet_to_be_applied -= _offset_value
                # Do accounting to republish total offset and others that were
                # set in the callbacks
                self.focus_offset_per_category["total"] = self.model.offset["z"]
            else:
                self.log.debug(
                    "Focus offset less than tolerance for "
                    "adjustment. Passing without correcting"
                )

            # publish events with new offsets, even if unchanged
            await self.evt_correctionOffsets.set_write(
                **self.model.offset, force_output=True
            )
            await self.evt_focusOffsetSummary.set_write(
                **self.focus_offset_per_category, force_output=True
            )

            # Now do pointing
            if np.max(np.abs(self.pointing_offsets_yet_to_be_applied)) > abs(
                _pointing_tolerance
            ):
                self.log.info(
                    f"Required pointing offset of [X,Y]={self.pointing_offsets_yet_to_be_applied} "
                    "arcsec in correction loop due to filter "
                    "and/or disperser changes."
                )
                _pointing_offsets = copy.deepcopy(
                    self.pointing_offsets_yet_to_be_applied
                )
            elif _offset_value:
                # No offsets above thresholds, so just return
                self.log.debug(
                    "Focus and pointing offsets less than tolerance required "
                    "for adjustment. Passing without correcting"
                )

            # Now apply any required corrections and publish events
            # This is required even if no focus or pointing changes are above
            # the thresholds

            # Corrections required, so flip the bit on the detailed state
            await self.set_detailed_state(np.uint8(self.detailed_state ^ status_bit))

            self.log.debug(
                f"_offset_value: {_offset_value} "
                f"_pointing_offsets: {_pointing_offsets!r}"
            )

            # send out even saying correction is started
            await self.evt_atspectrographCorrectionStarted.set_write(
                focusOffset=_offset_value,
                pointingOffsets=_pointing_offsets,
                force_output=True,
            )

            # Apply focus correction if above tolerance
            if abs(_offset_value) > 0.0:
                self.log.debug(
                    f"Applying focus correction {_offset_value}" " with hexapod"
                )
                await self.set_hexapod(azimuth, elevation)

            # Apply pointing correction if above tolerance
            # note that the comparison to zero should not be an issue as
            # it's explicitly set to zero then only changed if
            # the tolerance evaluation done above is satisfied.
            if np.max(abs(_pointing_offsets)) > 0:
                self.log.debug(
                    "Applying pointing correction _pointing_offsets"
                    f" = {_pointing_offsets} with atcs"
                )
                # apply offsets relative to what is already there
                await self.atcs.offset_xy(
                    _pointing_offsets[0],
                    _pointing_offsets[1],
                    relative=True,
                    persistent=True,
                )
                await asyncio.sleep(0)
                # update accounting and remove the already applied offsets
                self.pointing_offsets_per_category["total"] += _pointing_offsets
                self.pointing_offsets_yet_to_be_applied -= _pointing_offsets

                self.log.debug(
                    "new value of pointing_offsets_per_category['total'] is"
                    f" {self.pointing_offsets_per_category['total']}"
                )
            # Publish pointing summary
            await self.evt_pointingOffsetSummary.set_write(
                total=self.pointing_offsets_per_category["total"],
                filter=self.pointing_offsets_per_category["filter"],
                disperser=self.pointing_offsets_per_category["disperser"],
                force_output=True,
            )

            # Should only publish correction completed if no further
            # corrections are required
            # Add check to make sure it's not possible for the corrections
            # to be less than zero.
            if self.atspectrograph_corrections_required > 0:
                self.atspectrograph_corrections_required -= 1

            self.log.debug(
                "Corrections completed?"
                f"Pointing residuals zero?: "
                f"{np.max(np.abs(self.pointing_offsets_yet_to_be_applied)) <= 1e-12}, "
                f"Focus residuals zero? {abs(self.focus_offset_yet_to_be_applied) <= 1e-12}, "
                f"Required Correction loops remaining = {self.atspectrograph_corrections_required}"
            )
            self.log.debug(
                f"Pointing residuals are: {self.pointing_offsets_yet_to_be_applied}"
            )
            self.log.debug(
                f"Focus residuals are: {self.focus_offset_yet_to_be_applied}"
            )

            # Publish Event saying corrections are completed
            await self.evt_atspectrographCorrectionCompleted.set_write(
                focusOffset=_offset_value,
                pointingOffsets=_pointing_offsets,
                force_output=True,
            )
            # correction completed for this iteration,
            # flip bit on detailedState
            await self.set_detailed_state(np.uint8(self.detailed_state ^ status_bit))

    @staticmethod
    def get_config_pkg() -> str:
        return "ts_config_attcs"

    async def configure(self, config: typing.Any) -> None:
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

        self.correction_loop_time = 1.0 / config.correction_frequency
        self.log.debug(f"Correction loop time is {self.correction_loop_time}")

        for key in [
            "m1",
            "m2",
            "hexapod_x",
            "hexapod_y",
            "hexapod_z",
            "hexapod_u",
            "hexapod_v",
            "chromatic_dependence",
        ]:
            if hasattr(config, key):
                setattr(self.model, key, getattr(config, key))
            else:
                setattr(self.model, key, [0.0])

        if hasattr(config, "correction_tolerance"):
            for key in config.correction_tolerance:
                self.correction_tolerance[key] = config.correction_tolerance[key]

        self.model.hexapod_sensitivity_matrix = np.array(
            config.hexapod_sensitivity_matrix
        )

        self.model.m1_lut_elevation_limits = config.m1_lut_elevation_limits
        self.model.m1_pressure_minimum = config.m1_pressure_minimum
        self.model.m2_lut_elevation_limits = config.m2_lut_elevation_limits
        self.model.hexapod_lut_elevation_limits = config.hexapod_lut_elevation_limits

    def atspectrograph_summary_state_callback(
        self, data: type_hints.BaseMsgType
    ) -> None:
        """Callback to monitor summary state from atspectrograph. If this
        arises, then the filter/grating offsets that were previously set
        need to be removed as the new filter/grating values will come
        from events sent right afterwards when the component goes back
        to enabled state.

        Parameters
        ----------
        data : `ATSPectrograph_logevent_summaryState`
            Sample summaryState event from ATSpectrograph
        """

        self.atspectrograph_summary_state = State(data.summaryState)
        self.log.debug(
            f"Caught a new atspectrograph summary state {self.atspectrograph_summary_state!r}, "
            "resetting filter/disperser offsets"
        )
        # Add correction cycles so new corrections will be applied
        self.atspectrograph_corrections_required = 2  # one for focus, one for pointing
        # remove offsets from previous spectrograph setup
        self.focus_offset_yet_to_be_applied = -self.focus_offset_per_category["filter"]
        self.focus_offset_per_category["filter"] = 0.0

        self.focus_offset_yet_to_be_applied -= self.focus_offset_per_category[
            "disperser"
        ]
        self.focus_offset_per_category["disperser"] = 0.0

        self.focus_offset_yet_to_be_applied = -self.focus_offset_per_category[
            "wavelength"
        ]
        self.focus_offset_per_category["wavelength"] = 0.0

        self.pointing_offsets_yet_to_be_applied -= self.pointing_offsets_per_category[
            "filter"
        ]
        self.pointing_offsets_per_category["filter"] = np.array([0.0, 0.0])

        self.pointing_offsets_yet_to_be_applied -= self.pointing_offsets_per_category[
            "disperser"
        ]
        self.pointing_offsets_per_category["disperser"] = np.array([0.0, 0.0])

    async def check_atspectrograph(self) -> None:
        """Check that the atspectrograph is online and enabled.

        Raises
        ------
        RuntimeError
            If ATSpectrograph is not in ENABLED state.
        """
        if self.atspectrograph_summary_state != State.ENABLED:
            raise RuntimeError(
                f"ATSpectrograph (LATISS) in {self.atspectrograph_summary_state}. "
                f"Expected {State.ENABLED!r}. Enable "
                f"ATSpectrograph CSC before activating"
                f" corrections."
            )

    async def check_atpneumatic(self) -> None:
        """Check that ATPneumatics is in ENABLED state and that the main and
        instrument valves are open. If valves are closed try to open them.

        Raises
        ------
        RuntimeError
            If ATPneumatics is not in ENABLED state.
            If fails to open main or instrument valves.
        """

        if self.pneumatics_summary_state != State.ENABLED:
            raise RuntimeError(
                f"ATPneumatics in {self.pneumatics_summary_state}. "
                f"Expected {State.ENABLED!r}. Enable CSC before "
                "activating corrections."
            )

        if self.pneumatics_main_valve_state != ATPneumatics.AirValveState.OPENED:
            self.log.debug("ATPneumatics main valve not opened, trying to open it.")
            try:
                await self.pneumatics.cmd_openMasterAirSupply.start(
                    timeout=self.cmd_timeout
                )
            except AckError as ack_error:
                if ack_error.ackcmd.ack == SalRetCode.CMD_NOPERM:
                    self.log.exception("Master valve is already opened.")
                else:
                    raise RuntimeError(
                        "Failed to open ATPneumatics master air supply. "
                        f"ack: {ack_error.ackcmd.ack} error: {ack_error.ackcmd.result}"
                    )

        if self.pneumatics_instrument_valve_state != ATPneumatics.AirValveState.OPENED:
            self.log.debug(
                "ATPneumatics instrument valve not opened, trying to open it."
            )
            try:
                await self.pneumatics.cmd_openInstrumentAirValve.start(
                    timeout=self.cmd_timeout
                )
            except AckError as ack_error:
                if ack_error.ackcmd.ack == SalRetCode.CMD_NOPERM:
                    self.log.exception("Instrument valve is already opened.")
                else:
                    raise RuntimeError(
                        "Failed to open ATPneumatics instrument valve. "
                        f"ack: {ack_error.ackcmd.ack} error: {ack_error.ackcmd.result}"
                    )

    async def fault(
        self, code: typing.Optional[int] = None, report: str = "", traceback: str = ""
    ) -> None:
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
        traceback : `str` (optional)
            Error traceback string.
        """

        # Disable corrections
        disable_corr = self.cmd_disableCorrection.DataType()
        disable_corr.disableAll = True
        self.mark_corrections(disable_corr, False)

        await self.lower_mirrors_to_hardpoints(
            m1=self.corrections["m1"], m2=self.corrections["m2"]
        )

        await self.publish_enable_corrections()

        await super().fault(code=code, report=report, traceback=traceback)

    def pneumatics_summary_state_callback(self, data: type_hints.BaseMsgType) -> None:
        """Callback to monitor summary state from atpneumatics.

        Parameters
        ----------
        data : `ATPneumatics_logevent_summaryState`
            Sample summaryState event from ATPneumatics
        """
        self.pneumatics_summary_state = State(data.summaryState)

    def pneumatics_main_valve_state_callback(
        self, data: type_hints.BaseMsgType
    ) -> None:
        """Callback to monitor main valve state from atpneumatics.

        Parameters
        ----------
        data : `ATPneumatics_logevent_mainValveState`
            Sample mainValveState event from ATPneumatics
        """
        self.pneumatics_main_valve_state = ATPneumatics.AirValveState(data.state)

    def pneumatics_instrument_state_callback(
        self, data: type_hints.BaseMsgType
    ) -> None:
        """Callback to monitor instrument valve state from atpneumatics.

        Parameters
        ----------
        data : `ATPneumatics_logevent_instrumentState`
            Sample instrumentState event from ATPneumatics
        """
        self.pneumatics_instrument_valve_state = ATPneumatics.AirValveState(data.state)

    def pneumatics_m1_state_callback(self, data: type_hints.BaseMsgType) -> None:
        """Callback to monitor m1 valve state from atpneumatics.

        Parameters
        ----------
        data : `ATPneumatics_logevent_m1State`
            Sample m1State event from ATPneumatics
        """
        self.pneumatics_m1_state = ATPneumatics.AirValveState(data.state)

    def pneumatics_m2_state_callback(self, data: type_hints.BaseMsgType) -> None:
        """Callback to monitor m2 valve state from atpneumatics.

        Parameters
        ----------
        data : `ATPneumatics_logevent_m2State`
            Sample m2State event from ATPneumatics
        """
        self.pneumatics_m2_state = ATPneumatics.AirValveState(data.state)

    async def close(self) -> None:

        await self.atcs.close()
        await super().close()
        await self.camera.close()
        await self.atspectrograph.close()
