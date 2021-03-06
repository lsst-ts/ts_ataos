import unittest
from unittest.mock import Mock
import asyncio
import numpy as np
import logging
import pathlib

from lsst.ts import salobj

from lsst.ts.ataos import ataos_csc

from lsst.ts.idl.enums import ATPneumatics

index_gen = salobj.index_generator()

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.propagate = True

STD_TIMEOUT = 5  # standard command timeout (sec)
LONG_TIMEOUT = 20  # timeout for starting SAL components (sec)
TEST_CONFIG_DIR = pathlib.Path(__file__).parents[1].joinpath("tests", "data", "config")


class Harness:
    def __init__(self, config_dir=TEST_CONFIG_DIR):
        salobj.set_random_lsst_dds_partition_prefix()
        self.csc = ataos_csc.ATAOS(config_dir=config_dir)

        # Adds a remote to control the ATAOS CSC
        self.aos_remote = salobj.Remote(self.csc.domain, "ATAOS")

        # Adds Controllers to receive commands from the ATAOS system
        self.atmcs = salobj.Controller("ATMCS")
        self.atptg = salobj.Controller("ATPtg")
        self.pnematics = salobj.Controller("ATPneumatics")
        self.hexapod = salobj.Controller("ATHexapod")
        self.camera = salobj.Controller("ATCamera")
        self.atspectrograph = salobj.Controller("ATSpectrograph")

        # set the command timeout to be smaller so we don't have to wait for
        # errors. 5s is too short for enabling corrections.
        self.csc.cmd_timeout = 10.0

    async def __aenter__(self):
        await asyncio.gather(
            self.csc.start_task,
            self.aos_remote.start_task,
            self.atmcs.start_task,
            self.atptg.start_task,
            self.pnematics.start_task,
            self.hexapod.start_task,
            self.camera.start_task,
            self.atspectrograph.start_task,
        )
        return self

        # async def __aexit__(self, exc_type, exc_val, exc_tb):

    async def __aexit__(self, *args):
        await asyncio.gather(
            self.csc.close(),
            self.aos_remote.close(),
            self.atmcs.close(),
            self.atptg.close(),
            self.pnematics.close(),
            self.hexapod.close(),
            self.camera.close(),
            self.atspectrograph.close(),
        )


class TestCSC(unittest.TestCase):
    def test_standard_state_transitions(self):
        """Test standard CSC state transitions.

        The initial state is STANDBY.
        The standard commands and associated state transitions are:

        * enterControl: OFFLINE to STANDBY
        * start: STANDBY to DISABLED
        * enable: DISABLED to ENABLED

        * disable: ENABLED to DISABLED
        * standby: DISABLED to STANDBY
        * exitControl: STANDBY, FAULT to OFFLINE (quit)
        """

        async def doit():

            commands = ("start", "enable", "disable", "exitControl", "standby")

            extra_commands = (
                "applyFocusOffset",
                "disableCorrection",
                "enableCorrection",
            )

            async with Harness() as harness:

                # Define callbacks
                def callback(data):
                    pass

                harness.pnematics.cmd_m1SetPressure.callback = Mock(wraps=callback)
                harness.pnematics.cmd_m2SetPressure.callback = Mock(wraps=callback)

                # Check initial state
                current_state = await harness.aos_remote.evt_summaryState.next(
                    flush=False, timeout=1.0
                )

                self.assertEqual(harness.csc.summary_state, salobj.State.STANDBY)
                self.assertEqual(current_state.summaryState, salobj.State.STANDBY)

                # Check that settingVersions was published and matches
                # expected values
                setting_versions = await harness.aos_remote.evt_settingVersions.next(
                    flush=False, timeout=1.0
                )
                self.assertIsNotNone(setting_versions)

                for bad_command in commands:
                    if bad_command in ("start", "exitControl"):
                        continue  # valid command in STANDBY state
                    with self.subTest(bad_command=bad_command):
                        cmd_attr = getattr(harness.aos_remote, f"cmd_{bad_command}")
                        with self.assertRaises(salobj.AckError):
                            id_ack = await cmd_attr.start(
                                cmd_attr.DataType(), timeout=1.0
                            )

                for bad_command in extra_commands:
                    with self.subTest(bad_command=bad_command):
                        cmd_attr = getattr(harness.aos_remote, f"cmd_{bad_command}")
                        with self.assertRaises(salobj.AckError):
                            id_ack = await cmd_attr.start(
                                cmd_attr.DataType(), timeout=1.0
                            )

                # send start; new state is DISABLED
                cmd_attr = getattr(harness.aos_remote, "cmd_start")
                harness.aos_remote.evt_summaryState.flush()
                id_ack = await cmd_attr.start(
                    timeout=120
                )  # this one can take longer to execute
                state = await harness.aos_remote.evt_summaryState.next(
                    flush=False, timeout=5.0
                )
                self.assertEqual(id_ack.ack, salobj.SalRetCode.CMD_COMPLETE)
                self.assertEqual(id_ack.error, 0)
                self.assertEqual(harness.csc.summary_state, salobj.State.DISABLED)
                self.assertEqual(state.summaryState, salobj.State.DISABLED)
                # Verify mirror is NOT lowered when transitioning from STANDBY
                # to DISABLED -
                self.assertFalse(harness.pnematics.cmd_m1SetPressure.callback.called)
                self.assertFalse(harness.pnematics.cmd_m2SetPressure.callback.called)

                # TODO: There are two events issued when starting;
                # appliedSettingsMatchStart and settingsApplied.
                # Check that they are received.

                for bad_command in commands:
                    if bad_command in ("enable", "standby"):
                        continue  # valid command in DISABLED state
                    with self.subTest(bad_command=bad_command):
                        cmd_attr = getattr(harness.aos_remote, f"cmd_{bad_command}")
                        with self.assertRaises(salobj.AckError):
                            id_ack = await cmd_attr.start(
                                cmd_attr.DataType(), timeout=1.0
                            )

                for bad_command in extra_commands:
                    with self.subTest(bad_command=bad_command):
                        cmd_attr = getattr(harness.aos_remote, f"cmd_{bad_command}")
                        with self.assertRaises(salobj.AckError):
                            id_ack = await cmd_attr.start(
                                cmd_attr.DataType(), timeout=1.0
                            )

                # send enable; new state is ENABLED
                cmd_attr = getattr(harness.aos_remote, "cmd_enable")
                harness.aos_remote.evt_summaryState.flush()
                id_ack = await cmd_attr.start(
                    timeout=120
                )  # this one can take longer to execute
                state = await harness.aos_remote.evt_summaryState.next(
                    flush=False, timeout=5.0
                )
                self.assertEqual(id_ack.ack, salobj.SalRetCode.CMD_COMPLETE)
                self.assertEqual(id_ack.error, 0)
                self.assertEqual(harness.csc.summary_state, salobj.State.ENABLED)
                self.assertEqual(state.summaryState, salobj.State.ENABLED)

                for bad_command in commands:
                    if bad_command == "disable":
                        continue  # valid command in ENABLE state
                    with self.subTest(bad_command=bad_command):
                        cmd_attr = getattr(harness.aos_remote, f"cmd_{bad_command}")
                        with self.assertRaises(salobj.AckError):
                            id_ack = await cmd_attr.start(
                                cmd_attr.DataType(), timeout=1.0
                            )

                # Todo: Test that other commands works.
                # send disable; new state is DISABLED
                cmd_attr = getattr(harness.aos_remote, "cmd_disable")
                # this CMD may take some time to complete
                id_ack = await cmd_attr.start(cmd_attr.DataType(), timeout=30.0)
                self.assertEqual(id_ack.ack, salobj.SalRetCode.CMD_COMPLETE)
                self.assertEqual(id_ack.error, 0)
                self.assertEqual(harness.csc.summary_state, salobj.State.DISABLED)
                # Verify mirror is NOT lowered when transitioning from ENABLED
                # to DISABLED since corrections are not active
                self.assertFalse(harness.pnematics.cmd_m1SetPressure.callback.called)
                self.assertFalse(harness.pnematics.cmd_m2SetPressure.callback.called)

                # Bring to standby then enabled twice to verify bugfix as
                # part of DM-27243
                for i in range(2):
                    logger.debug(f"On iteration {i}, now enabling")
                    await salobj.set_summary_state(
                        harness.aos_remote, salobj.State.ENABLED, timeout=60
                    )
                    await asyncio.sleep(1)
                    logger.debug(f"On iteration {i}, now going to standby")
                    await salobj.set_summary_state(
                        harness.aos_remote, salobj.State.STANDBY, timeout=60
                    )

        asyncio.get_event_loop().run_until_complete(doit())

    def test_applyCorrection(self):
        """Test applyCorrection command. This commands applies the corrections
        for the current telescope position. It only works when the correction
        loop is not enabled.
        """

        async def doit(get_tel_pos=True, while_exposing=False):

            async with Harness() as harness:

                #
                # Check applyCorrection for position
                #
                def callback(data):
                    pass

                def hexapod_move_callback(data):
                    harness.hexapod.evt_positionUpdate.put()

                def m1_open_callback(data):
                    harness.pnematics.evt_m1State.set_put(
                        state=ATPneumatics.AirValveState.OPENED
                    )

                def m2_open_callback(data):
                    harness.pnematics.evt_m2State.set_put(
                        state=ATPneumatics.AirValveState.OPENED
                    )

                def m1_close_callback(data):
                    harness.pnematics.evt_m1State.set_put(
                        state=ATPneumatics.AirValveState.CLOSED
                    )

                def m2_close_callback(data):
                    harness.pnematics.evt_m2State.set_put(
                        state=ATPneumatics.AirValveState.CLOSED
                    )

                async def mount_offset_callback(self):
                    # just assume 0.1 degree offsets
                    harness.atmcs.evt_allAxesInPosition.set_put(inPosition=False)
                    await publish_mountEncoders(
                        harness, azimuth + 0.1, elevation + 0.1, ntimes=1
                    )
                    harness.atmcs.evt_allAxesInPosition.set_put(inPosition=True)

                # Set required callbacks
                harness.pnematics.cmd_m1SetPressure.callback = Mock(wraps=callback)
                harness.pnematics.cmd_m2SetPressure.callback = Mock(wraps=callback)
                harness.pnematics.cmd_m1OpenAirValve.callback = Mock(
                    wraps=m1_open_callback
                )
                harness.pnematics.cmd_m2OpenAirValve.callback = Mock(
                    wraps=m2_open_callback
                )
                harness.pnematics.cmd_m1CloseAirValve.callback = Mock(
                    wraps=m1_close_callback
                )
                harness.pnematics.cmd_m2CloseAirValve.callback = Mock(
                    wraps=m2_close_callback
                )

                harness.pnematics.evt_summaryState.set_put(
                    summaryState=salobj.State.ENABLED
                )
                harness.pnematics.evt_mainValveState.set_put(
                    state=ATPneumatics.AirValveState.OPENED
                )
                harness.pnematics.evt_instrumentState.set_put(
                    state=ATPneumatics.AirValveState.OPENED
                )
                harness.pnematics.evt_m1State.set_put(
                    state=ATPneumatics.AirValveState.CLOSED
                )
                harness.pnematics.evt_m2State.set_put(
                    state=ATPneumatics.AirValveState.CLOSED
                )

                harness.aos_remote.evt_detailedState.callback = Mock(wraps=callback)
                harness.atptg.cmd_poriginOffset.callback = Mock(
                    wraps=mount_offset_callback
                )

                # FIXME: Check if this is correct! Is there a difference in
                # command to move hexapod and focus or they will use the same
                # command?
                harness.hexapod.cmd_moveToPosition.callback = Mock(
                    wraps=hexapod_move_callback
                )

                # provide spectrograph setup info
                filter_name = "test_filt1"
                filter_focus_offset = 0.03
                filter_central_wavelength = 707.0
                filter_pointing_offsets = np.array([0.1, -0.1])

                disperser_name = "test_disp1"
                disperser_focus_offset = 0.1
                disperser_pointing_offsets = np.array([0.05, -0.05])

                # Bring spectrograph online and load filter/disperser
                harness.atspectrograph.evt_summaryState.set_put(
                    summaryState=salobj.State.ENABLED
                )

                harness.atspectrograph.evt_reportedFilterPosition.set_put(
                    name=filter_name,
                    centralWavelength=filter_central_wavelength,
                    focusOffset=filter_focus_offset,
                    pointingOffsets=filter_pointing_offsets,
                )
                harness.atspectrograph.evt_reportedDisperserPosition.set_put(
                    name=disperser_name,
                    focusOffset=disperser_focus_offset,
                    pointingOffsets=disperser_pointing_offsets,
                )

                logger.debug("Enabling ataos")
                await salobj.set_summary_state(harness.aos_remote, salobj.State.ENABLED)
                self.assertEqual(harness.csc.summary_state, salobj.State.ENABLED)
                logger.debug("Enabled ataos")

                logger.debug(
                    "Check that applyCorrection fails if enable Correction is on"
                )
                harness.aos_remote.cmd_enableCorrection.set(m1=True)
                await harness.aos_remote.cmd_enableCorrection.start(timeout=STD_TIMEOUT)

                with self.assertRaises(salobj.AckError):
                    await harness.aos_remote.cmd_applyCorrection.start(
                        timeout=STD_TIMEOUT
                    )

                logger.debug("Switch corrections off")
                harness.aos_remote.cmd_disableCorrection.set(disableAll=True)
                await harness.aos_remote.cmd_disableCorrection.start(
                    timeout=STD_TIMEOUT
                )

                logger.debug(
                    "Send applyCorrection command which will now "
                    "work because corrections are disabled"
                )
                azimuth = np.random.uniform(0.0, 360.0)
                # make sure it is never zero because np.random.uniform
                # is [min, max)
                elevation = 90.0 - np.random.uniform(0.0, 90.0)

                await publish_mountEncoders(harness, azimuth, elevation, ntimes=5)

                logger.debug(
                    "Test that the hexapod won't move if there's an exposure happening"
                )
                logger.debug(f"while_exposing is set to {while_exposing}")
                if while_exposing:
                    shutter_state_topic = (
                        harness.camera.evt_shutterDetailedState.DataType()
                    )
                    shutter_state_topic.substate = ataos_csc.ShutterState.OPEN
                    harness.camera.evt_shutterDetailedState.put(shutter_state_topic)
                    # Give some time for the CSC to grab that event
                    await asyncio.sleep(2.0)

                if not get_tel_pos:
                    await harness.aos_remote.cmd_applyCorrection.set_start(
                        azimuth=azimuth, elevation=elevation, timeout=STD_TIMEOUT
                    )
                else:
                    await harness.aos_remote.cmd_applyCorrection.start(
                        timeout=STD_TIMEOUT
                    )

                logger.debug("Check that callbacks where called")
                harness.pnematics.cmd_m1SetPressure.callback.assert_called()
                harness.pnematics.cmd_m2SetPressure.callback.assert_called()
                if while_exposing:
                    harness.hexapod.cmd_moveToPosition.callback.assert_not_called()
                    harness.atptg.cmd_poriginOffset.callback.assert_not_called()
                else:
                    harness.hexapod.cmd_moveToPosition.callback.assert_called()
                    harness.atptg.cmd_poriginOffset.callback.assert_called()

                harness.aos_remote.evt_detailedState.callback.assert_called()

                # Check that events where published with the correct az/el
                # position
                m1_start = await harness.aos_remote.evt_m1CorrectionStarted.next(
                    flush=False, timeout=STD_TIMEOUT
                )
                m2_start = await harness.aos_remote.evt_m2CorrectionStarted.next(
                    flush=False, timeout=STD_TIMEOUT
                )

                m1_end = await harness.aos_remote.evt_m1CorrectionCompleted.next(
                    flush=False, timeout=STD_TIMEOUT
                )
                m2_end = await harness.aos_remote.evt_m2CorrectionCompleted.next(
                    flush=False, timeout=STD_TIMEOUT
                )

                if while_exposing:
                    hx_start = None
                    hx_end = None
                    # hexapod should timeout
                    with self.assertRaises(asyncio.TimeoutError):
                        await harness.aos_remote.evt_hexapodCorrectionStarted.next(
                            flush=False, timeout=STD_TIMEOUT
                        )
                    with self.assertRaises(asyncio.TimeoutError):
                        await harness.aos_remote.evt_hexapodCorrectionCompleted.next(
                            flush=False, timeout=STD_TIMEOUT
                        )
                    # spectrograph should also timeout
                    with self.assertRaises(asyncio.TimeoutError):
                        await harness.aos_remote.evt_atspectrographCorrectionStarted.next(
                            flush=False, timeout=STD_TIMEOUT
                        )
                    with self.assertRaises(asyncio.TimeoutError):
                        await harness.aos_remote.evt_atspectrographCorrectionCompleted.next(
                            flush=False, timeout=STD_TIMEOUT
                        )
                else:
                    hx_start = (
                        await harness.aos_remote.evt_hexapodCorrectionStarted.next(
                            flush=False, timeout=STD_TIMEOUT
                        )
                    )
                    hx_end = (
                        await harness.aos_remote.evt_hexapodCorrectionCompleted.next(
                            flush=False, timeout=STD_TIMEOUT
                        )
                    )
                    ats_start = await harness.aos_remote.evt_atspectrographCorrectionStarted.next(
                        flush=False, timeout=STD_TIMEOUT
                    )
                    ats_end = await harness.aos_remote.evt_atspectrographCorrectionCompleted.next(
                        flush=False, timeout=STD_TIMEOUT
                    )
                    # pointing offsets only get published if they change, so
                    # only valid if the spectrograph is online, which is why
                    # it's inside the else statement
                    for component in (ats_start, ats_end):
                        if component is None:
                            continue
                        with self.subTest(component=component):
                            self.assertAlmostEqual(
                                component.focusOffset,
                                filter_focus_offset + disperser_focus_offset,
                            )
                        with self.subTest(component=component):
                            for n in range(len(component.pointingOffsets) - 1):
                                self.assertAlmostEqual(
                                    component.pointingOffsets[n],
                                    filter_pointing_offsets[n]
                                    + disperser_pointing_offsets[n],
                                )

                # Check that mirror and hexapod loops returned proper values
                for component in (m1_start, m2_start, hx_start, m1_end, m2_end, hx_end):
                    if component is None:
                        continue

                    with self.subTest(component=component, azimuth=azimuth):
                        self.assertEqual(component.azimuth, azimuth)
                    with self.subTest(component=component, elevation=elevation):
                        self.assertEqual(component.elevation, elevation)

                # Check how many times the detailed state bit was flipped
                # Should be a transition per component (2 flips), except
                # the spectrograph calls the hexapod, so it's actually
                # 4*2+2 = 10 detailed state transitions.
                # hexapod nor spectrograph function while exposing, so only
                # the mirrors should flip.
                self.assertEqual(
                    len(harness.aos_remote.evt_detailedState.callback.call_args_list),
                    10 if not while_exposing else 4,
                    "%s" % harness.aos_remote.evt_detailedState.callback.call_args_list,
                )

                logger.debug("Disable ATAOS CSC")
                await harness.aos_remote.cmd_disable.start(timeout=STD_TIMEOUT)

        # # Run test getting the telescope position
        asyncio.get_event_loop().run_until_complete(doit(get_tel_pos=True))
        logger.debug("test getting the telescope position - COMPLETE")

        # Run test getting the telescope position while exposing
        asyncio.get_event_loop().run_until_complete(
            doit(get_tel_pos=True, while_exposing=True)
        )
        logger.debug(
            "Run test getting the telescope position while exposing - COMPLETE"
        )

        # # Run for unspecified location
        asyncio.get_event_loop().run_until_complete(doit(get_tel_pos=False))
        logger.debug("Run for unspecified location - COMPLETE")

    def test_offsets(self):
        """Test offset command (which applies offsets to the models).
        It purposely does not test spectrograph offsets, which are
        handled in a subsequent test"""

        async def doit():

            async with Harness() as harness:

                # send pneumatics data
                harness.pnematics.evt_summaryState.set_put(
                    summaryState=salobj.State.ENABLED
                )
                harness.pnematics.evt_mainValveState.set_put(
                    state=ATPneumatics.AirValveState.OPENED
                )
                harness.pnematics.evt_instrumentState.set_put(
                    state=ATPneumatics.AirValveState.OPENED
                )
                harness.pnematics.evt_m1State.set_put(
                    state=ATPneumatics.AirValveState.OPENED
                )
                harness.pnematics.evt_m2State.set_put(
                    state=ATPneumatics.AirValveState.OPENED
                )

                # set the hexapod callback
                def hexapod_move_callback(data):
                    harness.hexapod.evt_positionUpdate.put()

                def m1_open_callback(data):
                    harness.pnematics.evt_m1State.set_put(
                        state=ATPneumatics.AirValveState.OPENED
                    )

                def m2_open_callback(data):
                    harness.pnematics.evt_m2State.set_put(
                        state=ATPneumatics.AirValveState.OPENED
                    )

                def m1_close_callback(data):
                    harness.pnematics.evt_m1State.set_put(
                        state=ATPneumatics.AirValveState.CLOSED
                    )

                def m2_close_callback(data):
                    harness.pnematics.evt_m2State.set_put(
                        state=ATPneumatics.AirValveState.CLOSED
                    )

                # Add callback to events
                def callback(data):
                    pass

                # Set required callbacks
                harness.hexapod.cmd_moveToPosition.callback = Mock(
                    wraps=hexapod_move_callback
                )
                harness.aos_remote.evt_detailedState.callback = Mock(wraps=callback)
                harness.pnematics.cmd_m1SetPressure.callback = Mock(wraps=callback)
                harness.pnematics.cmd_m2SetPressure.callback = Mock(wraps=callback)
                harness.pnematics.cmd_m1OpenAirValve.callback = Mock(
                    wraps=m1_open_callback
                )
                harness.pnematics.cmd_m2OpenAirValve.callback = Mock(
                    wraps=m2_open_callback
                )
                harness.pnematics.cmd_m1CloseAirValve.callback = Mock(
                    wraps=m1_close_callback
                )
                harness.pnematics.cmd_m2CloseAirValve.callback = Mock(
                    wraps=m2_close_callback
                )

                # Give a telescope position
                azimuth = 50.0
                elevation = 45.0
                await publish_mountEncoders(harness, azimuth, elevation, ntimes=1)
                harness.atmcs.evt_allAxesInPosition.set_put(inPosition=True)

                # if there is nothing (atpneumatics/atspectrograph) sending
                # events then this command times out need to extend the
                # timeout in this case.
                harness.aos_remote.evt_correctionOffsets.flush()

                await salobj.set_summary_state(
                    harness.aos_remote, salobj.State.ENABLED, timeout=60
                )
                self.assertEqual(harness.csc.summary_state, salobj.State.ENABLED)

                # Check that offset event was published on enable
                offset_init = await harness.aos_remote.evt_correctionOffsets.aget(
                    timeout=STD_TIMEOUT
                )

                offset = {
                    "m1": 1.0,
                    "m2": 1.0,
                    "x": 1.0,
                    "y": 1.0,
                    "z": 1.0,
                    "u": 1.0,
                    "v": 1.0,
                }

                for axis in offset:
                    with self.subTest(axis=axis):
                        self.assertEqual(0.0, getattr(offset_init, axis))

                logger.debug("Try to offset without loops closed, this should fail.")
                with self.assertRaises(salobj.AckError):
                    await harness.aos_remote.cmd_offset.set_start(
                        **offset, timeout=STD_TIMEOUT
                    )

                logger.debug("Now closing the loops")
                harness.aos_remote.evt_correctionOffsets.flush()
                await harness.aos_remote.cmd_enableCorrection.set_start(
                    m1=True, m2=True, hexapod=True, moveWhileExposing=False
                )

                logger.debug("Now sending offsets")
                await harness.aos_remote.cmd_offset.set_start(
                    **offset, timeout=STD_TIMEOUT
                )

                # grab most recent event
                offset_applied = await harness.aos_remote.evt_correctionOffsets.aget(
                    timeout=STD_TIMEOUT
                )

                for axis in offset:
                    with self.subTest(axis=axis):
                        self.assertEqual(offset[axis], getattr(offset_applied, axis))

                # Standard timeout sometimes isn't quite long enough for this,
                # so doubling it.
                logger.debug("Resetting Corrections")
                harness.aos_remote.evt_correctionOffsets.flush()
                await harness.aos_remote.cmd_resetOffset.start(timeout=STD_TIMEOUT * 2)

                offset_reset = await harness.aos_remote.evt_correctionOffsets.aget(
                    timeout=LONG_TIMEOUT
                )

                for axis in offset:
                    with self.subTest(axis=axis):
                        self.assertEqual(0.0, getattr(offset_reset, axis))

                # Open two loops, mostly so the output is cleaner
                logger.debug("Disabling hexapod and spectrograph corrections")
                await harness.aos_remote.cmd_disableCorrection.set_start(
                    hexapod=True, timeout=STD_TIMEOUT
                )

                # Send to disabled state
                await harness.aos_remote.cmd_disable.start()

                # Verify mirror is lowered when transitioning from ENABLED
                # to DISABLED since corrections are enabled
                # this is done by sending a pressure of zero

                self.assertEqual(
                    (harness.pnematics.cmd_m1SetPressure.callback.call_args[0])[
                        0
                    ].pressure,
                    0.0,
                )
                self.assertEqual(
                    (harness.pnematics.cmd_m2SetPressure.callback.call_args[0])[
                        0
                    ].pressure,
                    0.0,
                )

        # # Run for unspecified location
        asyncio.get_event_loop().run_until_complete(doit())

    def test_spectrograph_offsets(self):
        """Test offsets command and handling of offsets during filter/grating
        changes."""

        async def doit(atspectrograph=True, online_before_ataos=False):

            async with Harness() as harness:

                # send pneumatics data, just speeds up the tests
                harness.pnematics.evt_summaryState.set_put(
                    summaryState=salobj.State.ENABLED
                )
                harness.pnematics.evt_mainValveState.set_put(
                    state=ATPneumatics.AirValveState.OPENED
                )
                harness.pnematics.evt_instrumentState.set_put(
                    state=ATPneumatics.AirValveState.OPENED
                )
                harness.pnematics.evt_m1State.set_put(
                    state=ATPneumatics.AirValveState.CLOSED
                )
                harness.pnematics.evt_m2State.set_put(
                    state=ATPneumatics.AirValveState.CLOSED
                )

                # set the hexapod callback
                def hexapod_move_callback(data):
                    harness.hexapod.evt_positionUpdate.put()

                # Add callback to events
                def callback(data):
                    pass

                async def mount_offset_callback(self):
                    # just assume 0.1 degree offets
                    harness.atmcs.evt_allAxesInPosition.set_put(inPosition=False)
                    await publish_mountEncoders(
                        harness, azimuth + 0.1, elevation + 0.1, ntimes=1
                    )
                    harness.atmcs.evt_allAxesInPosition.set_put(inPosition=True)

                # Set required callbacks
                harness.hexapod.cmd_moveToPosition.callback = Mock(
                    wraps=hexapod_move_callback
                )
                harness.aos_remote.evt_detailedState.callback = Mock(wraps=callback)
                harness.atptg.cmd_poriginOffset.callback = Mock(
                    wraps=mount_offset_callback
                )

                # Can only set these if the spectrograph is not going to come
                # online if it goes offline the values will remain and no new
                # filter/disperser positions will get published
                if atspectrograph:
                    filter_name, filter_name2 = "test_filt1", "test_filt2"
                    filter_focus_offset, filter_focus_offset2 = 0.03, 0.0
                    filter_central_wavelength, filter_central_wavelength2 = 707.0, 700
                    filter_pointing_offsets = np.array([0.1, 0.0])
                    filter_pointing_offsets2 = np.array([0.0, -0.2])

                    disperser_name, disperser_name2 = "test_disp1", "test_disp2"
                    disperser_focus_offset, disperser_focus_offset2 = 0.1, 0.0
                    disperser_pointing_offsets = np.array([0.0, -0.05])
                    disperser_pointing_offsets2 = np.array([0.3, 0.0])
                else:
                    # These are needed for testing purposes when the
                    # spectrograph is offline.
                    filter_name, filter_name2 = "", ""
                    filter_focus_offset, filter_focus_offset2 = 0.0, 0.0
                    filter_central_wavelength, filter_central_wavelength2 = 0.0, 0.0

                    disperser_name, disperser_name2 = "", ""
                    disperser_focus_offset, disperser_focus_offset2 = 0.0, 0.0
                    disperser_pointing_offsets = np.array([0.0, 0.0])
                    disperser_pointing_offsets2 = np.array([0.0, 0.0])

                if atspectrograph and online_before_ataos:
                    logger.debug("Loading filter and dispersers before enabling ATAOS")
                    # Bring spectrograph online and load filter/disperser
                    harness.atspectrograph.evt_summaryState.set_put(
                        summaryState=salobj.State.ENABLED
                    )

                    harness.atspectrograph.evt_reportedFilterPosition.set_put(
                        name=filter_name,
                        centralWavelength=filter_central_wavelength,
                        focusOffset=filter_focus_offset,
                        pointingOffsets=filter_pointing_offsets,
                    )
                    harness.atspectrograph.evt_reportedDisperserPosition.set_put(
                        name=disperser_name,
                        focusOffset=disperser_focus_offset,
                        pointingOffsets=disperser_pointing_offsets,
                    )

                # If the atpneumatics and atspectrograph sending events then
                # this command times out when using the default timeout,
                # therefore it is extended here to account for that case.

                await salobj.set_summary_state(
                    harness.aos_remote, salobj.State.ENABLED, timeout=60
                )
                self.assertEqual(harness.csc.summary_state, salobj.State.ENABLED)

                # send elevation/azimuth positions
                azimuth = np.random.uniform(0.0, 360.0)
                # make sure it is never zero because np.random.uniform is
                # [min, max)
                elevation = 90.0 - np.random.uniform(0.0, 90.0)

                await publish_mountEncoders(harness, azimuth, elevation, ntimes=5)

                # Bring the spectrograph online if not already
                if atspectrograph and not online_before_ataos:
                    logger.debug("Loading filter and dispersers after enabling ATAOS")
                    # Bring spectrograph online and load filter/disperser
                    harness.atspectrograph.evt_summaryState.set_put(
                        summaryState=salobj.State.ENABLED
                    )

                    # the summarystate causes offsets to be published, so
                    # flush these to be sure the grab the proper one below
                    harness.aos_remote.evt_correctionOffsets.flush()
                    harness.aos_remote.evt_focusOffsetSummary.flush()
                    harness.aos_remote.evt_atspectrographCorrectionStarted.flush()
                    harness.aos_remote.evt_atspectrographCorrectionCompleted.flush()

                    harness.atspectrograph.evt_reportedFilterPosition.set_put(
                        name=filter_name,
                        centralWavelength=filter_central_wavelength,
                        focusOffset=filter_focus_offset,
                        pointingOffsets=filter_pointing_offsets,
                    )
                    harness.atspectrograph.evt_reportedDisperserPosition.set_put(
                        name=disperser_name,
                        focusOffset=disperser_focus_offset,
                        pointingOffsets=disperser_pointing_offsets,
                    )

                logger.debug(
                    "Make sure spectrograph corrections have not "
                    "been applied, so this should fail"
                )
                with self.assertRaises(asyncio.TimeoutError):
                    await harness.aos_remote.evt_atspectrographCorrectionStarted.next(
                        timeout=STD_TIMEOUT, flush=False
                    )
                with self.assertRaises(asyncio.TimeoutError):
                    await harness.aos_remote.evt_atspectrographCorrectionCompleted.next(
                        timeout=STD_TIMEOUT, flush=False
                    )

                # Turn on spectrograph corrections
                if atspectrograph:
                    logger.debug("Enabling atspectrograph and hexapod corrections")
                    await harness.aos_remote.cmd_enableCorrection.set_start(
                        atspectrograph=True, hexapod=True
                    )

                    pointingOffsetSummary = (
                        await harness.aos_remote.evt_pointingOffsetSummary.next(
                            flush=False, timeout=STD_TIMEOUT * 2
                        )
                    )
                    # check atspectrograph corrections were applied
                    await harness.aos_remote.evt_atspectrographCorrectionCompleted.next(
                        timeout=STD_TIMEOUT, flush=False
                    )
                    await harness.aos_remote.evt_atspectrographCorrectionStarted.next(
                        timeout=STD_TIMEOUT, flush=False
                    )

                # check corrections were applied
                correctionOffsets = await harness.aos_remote.evt_correctionOffsets.aget(
                    timeout=STD_TIMEOUT
                )
                # check focus accounting is being done correctly
                focusOffsetSummary = (
                    await harness.aos_remote.evt_focusOffsetSummary.aget(
                        timeout=STD_TIMEOUT
                    )
                )

                if atspectrograph:
                    self.assertAlmostEqual(
                        correctionOffsets.z,
                        filter_focus_offset + disperser_focus_offset,
                    )
                    self.assertAlmostEqual(
                        focusOffsetSummary.total,
                        disperser_focus_offset + filter_focus_offset,
                    )
                    # Check pointingOffsets is correct, but note they are
                    # arrays. Must loop over values individually
                    for n in range(len(pointingOffsetSummary.total) - 1):
                        self.assertAlmostEqual(
                            pointingOffsetSummary.filter[n], filter_pointing_offsets[n]
                        )
                        self.assertAlmostEqual(
                            pointingOffsetSummary.disperser[n],
                            disperser_pointing_offsets[n],
                        )
                        self.assertAlmostEqual(
                            pointingOffsetSummary.total[n],
                            filter_pointing_offsets[n] + disperser_pointing_offsets[n],
                        )
                    logger.debug("Disabling corrections")
                    await harness.aos_remote.cmd_disableCorrection.set_start(
                        hexapod=True, atspectrograph=True
                    )

                else:
                    # offsets from filter/dispersers should not yet be
                    # applied if correction loop for spectrograph isn't on
                    self.assertAlmostEqual(correctionOffsets.z, 0.0)
                    self.assertAlmostEqual(focusOffsetSummary.total, 0.0)

                self.assertAlmostEqual(focusOffsetSummary.userApplied, 0.0)
                self.assertAlmostEqual(focusOffsetSummary.filter, filter_focus_offset)
                self.assertAlmostEqual(
                    focusOffsetSummary.disperser, disperser_focus_offset
                )

                offset = {
                    "m1": 0.0,
                    "m2": 0.0,
                    "x": 1.3,
                    "y": 1.4,
                    "z": 1.5,
                    "u": 1.6,
                    "v": 1.7,
                }

                offset2 = {
                    "m1": 1.0,
                    "m2": 2.0,
                    "x": 2.3,
                    "y": 2.4,
                    "z": 2.5,
                    "u": 2.6,
                    "v": 2.7,
                }

                # Now start the loops, we'll then add an offset,
                # then remove a filter, then remove a disperser
                if atspectrograph:
                    logger.debug("Re-enabling atspectrograph and hexapod corrections")
                    await harness.aos_remote.cmd_enableCorrection.set_start(
                        atspectrograph=True, hexapod=True
                    )

                # Wait for corrections to be applied
                await asyncio.sleep(4)
                logger.debug(
                    "Try to apply an offset without the correction on, this should fail."
                )
                with self.assertRaises(salobj.AckError):
                    await harness.aos_remote.cmd_offset.set_start(
                        **offset2, timeout=STD_TIMEOUT
                    )

                offset_applied = await harness.aos_remote.evt_correctionOffsets.aget()
                focusOffsetSummary = (
                    await harness.aos_remote.evt_focusOffsetSummary.aget()
                )

                logger.debug(
                    f"Before offset, focusOffsetSummary is {focusOffsetSummary}"
                )
                logger.debug(f"Before offset, offset_applied is {offset_applied}")

                # flush events then send relative offsets
                harness.aos_remote.evt_correctionOffsets.flush()
                harness.aos_remote.evt_focusOffsetSummary.flush()

                # add the userApplied-offset
                await harness.aos_remote.cmd_offset.set_start(
                    **offset, timeout=STD_TIMEOUT
                )

                offset_applied = await harness.aos_remote.evt_correctionOffsets.next(
                    flush=False, timeout=STD_TIMEOUT
                )
                focusOffsetSummary = (
                    await harness.aos_remote.evt_focusOffsetSummary.next(
                        flush=False, timeout=STD_TIMEOUT
                    )
                )

                # offsets should be combined in z
                for axis in offset:
                    logger.debug(
                        "axis = {} and correction_offset"
                        " = {}".format(axis, getattr(offset_applied, axis))
                    )
                    with self.subTest(axis=axis):
                        if axis != "z":
                            self.assertAlmostEqual(
                                offset[axis], getattr(offset_applied, axis)
                            )
                        else:
                            # should be applied offset plus the
                            # filter/disperser offsets
                            self.assertAlmostEqual(
                                offset[axis]
                                + disperser_focus_offset
                                + filter_focus_offset,
                                getattr(offset_applied, axis),
                            )

                # check that summary is correct
                self.assertAlmostEqual(
                    focusOffsetSummary.total, getattr(offset_applied, "z")
                )
                # userApplied offset should just be whatever we supplied
                logger.debug(
                    f"Checking assertions. focusOffsetSummary is {focusOffsetSummary}"
                )
                logger.debug(f"Checking assertions. offset_applied is {offset_applied}")
                self.assertAlmostEqual(focusOffsetSummary.userApplied, offset["z"])
                self.assertAlmostEqual(focusOffsetSummary.filter, filter_focus_offset)
                self.assertAlmostEqual(
                    focusOffsetSummary.disperser, disperser_focus_offset
                )

                # This part of the test is only applicable if the spectrograph
                # is online
                if atspectrograph:
                    logger.debug("Putting in filter2")
                    # flush events then change filters
                    harness.aos_remote.evt_correctionOffsets.flush()
                    harness.aos_remote.evt_focusOffsetSummary.flush()
                    harness.aos_remote.evt_pointingOffsetSummary.flush()
                    harness.aos_remote.evt_atspectrographCorrectionStarted.flush()
                    harness.aos_remote.evt_atspectrographCorrectionCompleted.flush()

                    harness.atspectrograph.evt_reportedFilterPosition.set_put(
                        name=filter_name2,
                        centralWavelength=filter_central_wavelength2,
                        focusOffset=filter_focus_offset2,
                        pointingOffsets=filter_pointing_offsets2,
                    )

                    await harness.aos_remote.evt_atspectrographCorrectionStarted.next(
                        timeout=STD_TIMEOUT, flush=False
                    )
                    await harness.aos_remote.evt_atspectrographCorrectionCompleted.next(
                        timeout=STD_TIMEOUT, flush=False
                    )
                    # check pointing offset was applied
                    harness.atptg.cmd_poriginOffset.callback.assert_called()

                    # check focus model updates were applied
                    offset_applied = (
                        await harness.aos_remote.evt_correctionOffsets.next(
                            flush=False, timeout=STD_TIMEOUT
                        )
                    )
                    focusOffsetSummary = (
                        await harness.aos_remote.evt_focusOffsetSummary.next(
                            flush=False, timeout=STD_TIMEOUT
                        )
                    )
                    pointingOffsetSummary = (
                        await harness.aos_remote.evt_pointingOffsetSummary.next(
                            flush=False, timeout=STD_TIMEOUT
                        )
                    )

                    # offsets should be combined in z
                    # this could be made a function, but I found this easier
                    # to read/parse/understand
                    for axis in offset:

                        with self.subTest(axis=axis):
                            if axis != "z":
                                self.assertAlmostEqual(
                                    offset[axis], getattr(offset_applied, axis)
                                )
                            else:
                                # should be applied offset plus the
                                # filter/disperser offsets
                                self.assertAlmostEqual(
                                    offset[axis]
                                    + disperser_focus_offset
                                    + filter_focus_offset2,
                                    getattr(offset_applied, axis),
                                )

                    # check that focus summary is correct
                    self.assertAlmostEqual(
                        focusOffsetSummary.total, getattr(offset_applied, "z")
                    )
                    # userApplied offset should just be whatever we supplied
                    self.assertAlmostEqual(focusOffsetSummary.userApplied, offset["z"])
                    self.assertAlmostEqual(
                        focusOffsetSummary.filter, filter_focus_offset2
                    )
                    self.assertAlmostEqual(
                        focusOffsetSummary.disperser, disperser_focus_offset
                    )

                    # Check pointingOffsets is correct, but note they are
                    # arrays. Must loop over values individually
                    for n in range(len(pointingOffsetSummary.total) - 1):
                        self.assertAlmostEqual(
                            pointingOffsetSummary.filter[n], filter_pointing_offsets2[n]
                        )
                        self.assertAlmostEqual(
                            pointingOffsetSummary.disperser[n],
                            disperser_pointing_offsets[n],
                        )
                        self.assertAlmostEqual(
                            pointingOffsetSummary.total[n],
                            filter_pointing_offsets2[n] + disperser_pointing_offsets[n],
                        )

                    # flush events then change dispersers
                    logger.debug("Putting in disperser2")
                    harness.aos_remote.evt_correctionOffsets.flush()
                    harness.aos_remote.evt_focusOffsetSummary.flush()
                    harness.aos_remote.evt_pointingOffsetSummary.flush()
                    harness.aos_remote.evt_atspectrographCorrectionStarted.flush()
                    harness.aos_remote.evt_atspectrographCorrectionCompleted.flush()

                    harness.atspectrograph.evt_reportedDisperserPosition.set_put(
                        name=disperser_name2,
                        focusOffset=disperser_focus_offset2,
                        pointingOffsets=disperser_pointing_offsets2,
                    )

                    await harness.aos_remote.evt_atspectrographCorrectionStarted.next(
                        timeout=STD_TIMEOUT, flush=False
                    )
                    await harness.aos_remote.evt_atspectrographCorrectionCompleted.next(
                        timeout=STD_TIMEOUT, flush=False
                    )
                    # check pointing offset was applied
                    harness.atptg.cmd_poriginOffset.callback.assert_called()

                    offset_applied = (
                        await harness.aos_remote.evt_correctionOffsets.next(
                            flush=False, timeout=STD_TIMEOUT
                        )
                    )
                    focusOffsetSummary = (
                        await harness.aos_remote.evt_focusOffsetSummary.next(
                            flush=False, timeout=STD_TIMEOUT
                        )
                    )
                    pointingOffsetSummary = (
                        await harness.aos_remote.evt_pointingOffsetSummary.next(
                            flush=False, timeout=STD_TIMEOUT
                        )
                    )

                    # offsets should be combined in z
                    for axis in offset:

                        with self.subTest(axis=axis):
                            if axis != "z":
                                self.assertAlmostEqual(
                                    offset[axis], getattr(offset_applied, axis)
                                )
                            else:
                                # should be applied offset plus the
                                # filter/disperser offsets
                                self.assertAlmostEqual(
                                    offset[axis]
                                    + disperser_focus_offset2
                                    + filter_focus_offset2,
                                    getattr(offset_applied, axis),
                                )

                    # check that summary is correct
                    self.assertAlmostEqual(
                        focusOffsetSummary.total, getattr(offset_applied, "z")
                    )
                    # userApplied offset should just be whatever we supplied
                    self.assertAlmostEqual(focusOffsetSummary.userApplied, offset["z"])
                    self.assertAlmostEqual(
                        focusOffsetSummary.filter, filter_focus_offset2
                    )
                    self.assertAlmostEqual(
                        focusOffsetSummary.disperser, disperser_focus_offset2
                    )

                    # Check pointingOffsets is correct, but note they are
                    # arrays. Must loop over values individually
                    for n in range(len(pointingOffsetSummary.total) - 1):
                        self.assertAlmostEqual(
                            pointingOffsetSummary.filter[n], filter_pointing_offsets2[n]
                        )
                        self.assertAlmostEqual(
                            pointingOffsetSummary.disperser[n],
                            disperser_pointing_offsets2[n],
                        )
                        self.assertAlmostEqual(
                            pointingOffsetSummary.total[n],
                            filter_pointing_offsets2[n]
                            + disperser_pointing_offsets2[n],
                        )

                # Now reset the offsets (after flushing events)
                # will not reset spectrograph offsets!
                # because not all loops are enabled we have to do this
                # one at a time.
                harness.aos_remote.evt_correctionOffsets.flush()
                harness.aos_remote.evt_focusOffsetSummary.flush()

                logger.debug(
                    "Try to reset an offset without the correction on, this should fail."
                )
                with self.assertRaises(salobj.AckError):
                    await harness.aos_remote.cmd_resetOffset.set_start(
                        axis="m1", timeout=STD_TIMEOUT
                    )

                hexapod_axes = ["x", "y", "z", "u", "v"]
                for axis in hexapod_axes:
                    await harness.aos_remote.cmd_resetOffset.set_start(
                        axis=axis, timeout=STD_TIMEOUT
                    )

                offset_applied = await harness.aos_remote.evt_correctionOffsets.aget(
                    timeout=STD_TIMEOUT
                )
                focusOffsetSummary = (
                    await harness.aos_remote.evt_focusOffsetSummary.aget(
                        timeout=STD_TIMEOUT
                    )
                )

                for axis in offset:
                    with self.subTest(axis=axis):
                        if axis != "z":
                            self.assertAlmostEqual(0.0, getattr(offset_applied, axis))
                        else:
                            # correction offset should be zero plus the
                            # filter/disperser offsets
                            self.assertAlmostEqual(
                                0.0 + disperser_focus_offset2 + filter_focus_offset2,
                                getattr(offset_applied, axis),
                            )
                # totals should be just filter/disperser offsets
                self.assertAlmostEqual(
                    focusOffsetSummary.total,
                    disperser_focus_offset2 + filter_focus_offset2,
                )
                # userApplied offset should be zero
                self.assertAlmostEqual(focusOffsetSummary.userApplied, 0.0)
                self.assertAlmostEqual(focusOffsetSummary.filter, filter_focus_offset2)
                self.assertAlmostEqual(
                    focusOffsetSummary.disperser, disperser_focus_offset2
                )

                if atspectrograph:
                    # Verify that no spectrograph specific offset took place
                    logger.debug(
                        "Make sure spectrograph correction is not "
                        "applied, so this should fail"
                    )
                    with self.assertRaises(asyncio.TimeoutError):
                        await harness.aos_remote.evt_atspectrographCorrectionStarted.next(
                            timeout=STD_TIMEOUT, flush=False
                        )
                    with self.assertRaises(asyncio.TimeoutError):
                        await harness.aos_remote.evt_atspectrographCorrectionCompleted.next(
                            timeout=STD_TIMEOUT, flush=False
                        )
                    # Check pointingOffsets is correct, but note they are
                    # arrays. Must loop over values individually
                    for n in range(len(pointingOffsetSummary.total) - 1):
                        self.assertAlmostEqual(
                            pointingOffsetSummary.filter[n], filter_pointing_offsets2[n]
                        )
                        self.assertAlmostEqual(
                            pointingOffsetSummary.disperser[n],
                            disperser_pointing_offsets2[n],
                        )
                        self.assertAlmostEqual(
                            pointingOffsetSummary.total[n],
                            filter_pointing_offsets2[n]
                            + disperser_pointing_offsets2[n],
                        )

                # Disable corrections gracefully, makes debugging easier
                logger.debug("Disabling corrections")
                await harness.aos_remote.cmd_disableCorrection.set_start(
                    hexapod=True, atspectrograph=True
                )

                # send spectrograph offline
                harness.atspectrograph.evt_summaryState.set_put(
                    summaryState=salobj.State.OFFLINE
                )

        logger.debug("Running test with spectrograph online before ATAOS")
        asyncio.get_event_loop().run_until_complete(
            doit(atspectrograph=True, online_before_ataos=True)
        )
        logger.debug("COMPLETED test with spectrograph online before ATAOS \n")

        logger.debug("Running test with spectrograph online after ATAOS")
        asyncio.get_event_loop().run_until_complete(
            doit(atspectrograph=True, online_before_ataos=False)
        )
        logger.debug("COMPLETED test with spectrograph online after ATAOS \n")

    def test_spectrograph_wavelength_offsets(self):
        """Test offsets command and handling of wavelength offsets during
        filter/grating changes."""

        async def doit():

            async with Harness() as harness:

                # send pneumatics data, just speeds up the tests
                harness.pnematics.evt_summaryState.set_put(
                    summaryState=salobj.State.ENABLED
                )
                harness.pnematics.evt_mainValveState.set_put(
                    state=ATPneumatics.AirValveState.OPENED
                )
                harness.pnematics.evt_instrumentState.set_put(
                    state=ATPneumatics.AirValveState.OPENED
                )
                harness.pnematics.evt_m1State.set_put(
                    state=ATPneumatics.AirValveState.CLOSED
                )
                harness.pnematics.evt_m2State.set_put(
                    state=ATPneumatics.AirValveState.CLOSED
                )

                # set the hexapod callback
                def hexapod_move_callback(data):
                    harness.hexapod.evt_positionUpdate.put()

                # Add callback to events
                def callback(data):
                    pass

                async def mount_offset_callback(self):
                    # just assume 0.1 degree offets
                    harness.atmcs.evt_allAxesInPosition.set_put(inPosition=False)
                    await publish_mountEncoders(
                        harness, azimuth + 0.1, elevation + 0.1, ntimes=1
                    )
                    harness.atmcs.evt_allAxesInPosition.set_put(inPosition=True)

                # Set required callbacks
                harness.hexapod.cmd_moveToPosition.callback = Mock(
                    wraps=hexapod_move_callback
                )
                harness.aos_remote.evt_detailedState.callback = Mock(wraps=callback)
                harness.atptg.cmd_poriginOffset.callback = Mock(
                    wraps=mount_offset_callback
                )

                # Can only set these if the spectrograph is not going to come
                # online if it goes offline the values will remain and no new
                # filter/disperser positions will get published

                filter_name, filter_name2 = "test_filt1", "test_filt2"
                filter_focus_offset, filter_focus_offset2 = 0.03, 0.0
                filter_central_wavelength, filter_central_wavelength2 = 707.0, 700
                filter_pointing_offsets = np.array([0.1, -0.1])
                filter_pointing_offsets2 = np.array([0.2, -0.2])

                disperser_name, disperser_name2 = "test_disp1", "test_disp2"
                disperser_focus_offset, disperser_focus_offset2 = 0.1, 0.0
                disperser_pointing_offsets = np.array([0.05, -0.05])
                disperser_pointing_offsets2 = np.array([0.13, -0.13])

                logger.debug("Loading filter and dispersers before enabling ATAOS")
                # Bring spectrograph online and load filter/disperser
                harness.atspectrograph.evt_summaryState.set_put(
                    summaryState=salobj.State.ENABLED
                )

                harness.atspectrograph.evt_reportedFilterPosition.set_put(
                    name=filter_name,
                    centralWavelength=filter_central_wavelength,
                    focusOffset=filter_focus_offset,
                    pointingOffsets=filter_pointing_offsets,
                )
                harness.atspectrograph.evt_reportedDisperserPosition.set_put(
                    name=disperser_name,
                    focusOffset=disperser_focus_offset,
                    pointingOffsets=disperser_pointing_offsets,
                )
                logger.debug("Awaiting here for 1s, then enabling ATAOS")
                await asyncio.sleep(1)

                # If the atpneumatics and atspectrograph sending events then
                # this command times out when using the default timeout,
                # therefore it is extended here to account for that case.

                await salobj.set_summary_state(
                    harness.aos_remote,
                    salobj.State.ENABLED,
                    settingsToApply="current",
                    timeout=60,
                )

                logger.debug(
                    "Make sure spectrograph correction is not "
                    "applied, so this should fail"
                )
                with self.assertRaises(asyncio.TimeoutError):
                    await harness.aos_remote.evt_atspectrographCorrectionStarted.next(
                        timeout=STD_TIMEOUT, flush=False
                    )
                with self.assertRaises(asyncio.TimeoutError):
                    await harness.aos_remote.evt_atspectrographCorrectionCompleted.next(
                        timeout=STD_TIMEOUT, flush=False
                    )

                self.assertEqual(harness.csc.summary_state, salobj.State.ENABLED)

                # send elevation/azimuth positions
                azimuth = np.random.uniform(0.0, 360.0)
                # make sure it is never zero because np.random.uniform is
                # [min, max)
                elevation = 90.0 - np.random.uniform(0.0, 90.0)

                await publish_mountEncoders(harness, azimuth, elevation, ntimes=5)

                logger.debug(
                    "Send a new central wavelength without closing the loops, this should fail."
                )
                new_cen_wave = 500
                with self.assertRaises(salobj.AckError):
                    await harness.aos_remote.cmd_setWavelength.set_start(
                        wavelength=new_cen_wave
                    )

                logger.debug(
                    "Try to add a manual focus offset with the loops off, this should fail."
                )
                with self.assertRaises(salobj.AckError):
                    await harness.aos_remote.cmd_applyFocusOffset.set_start(
                        offset=1.0, timeout=STD_TIMEOUT
                    )

                # Turn on appropriate corrections
                logger.debug("Enabling atspectrograph and hexapod corrections")
                await harness.aos_remote.cmd_enableCorrection.set_start(
                    atspectrograph=True, hexapod=True
                )

                logger.debug("Send new central wavelength with loop closed.")
                harness.aos_remote.evt_correctionOffsets.flush()
                harness.aos_remote.evt_focusOffsetSummary.flush()
                harness.aos_remote.evt_atspectrographCorrectionStarted.flush()
                harness.aos_remote.evt_atspectrographCorrectionCompleted.flush()

                await harness.aos_remote.cmd_setWavelength.set_start(
                    wavelength=new_cen_wave
                )

                # evaluate expectation from value in configuration
                # based on value 2.1e-5 mm/nm, offset should be ~0.0042mm

                # central wavelength of the telescope with no filter is 700nm
                focus_wave_expect = np.poly1d(
                    harness.csc.model.config["chromatic_dependence"]
                )(new_cen_wave - 700)
                logger.debug(
                    f"Calculated focus offset due to wavelength dependence is {focus_wave_expect}"
                )

                # check corrections were applied
                correctionOffsets = await harness.aos_remote.evt_correctionOffsets.aget(
                    timeout=STD_TIMEOUT
                )
                # check correction started/completed events were sent
                await harness.aos_remote.evt_atspectrographCorrectionStarted.next(
                    timeout=STD_TIMEOUT, flush=False
                )
                await harness.aos_remote.evt_atspectrographCorrectionCompleted.next(
                    timeout=STD_TIMEOUT, flush=False
                )
                # check focus accounting is being done correctly
                focusOffsetSummary = (
                    await harness.aos_remote.evt_focusOffsetSummary.aget(
                        timeout=STD_TIMEOUT
                    )
                )

                self.assertAlmostEqual(focusOffsetSummary.wavelength, focus_wave_expect)
                self.assertAlmostEqual(
                    correctionOffsets.z,
                    filter_focus_offset + disperser_focus_offset + focus_wave_expect,
                )
                self.assertAlmostEqual(
                    focusOffsetSummary.total,
                    disperser_focus_offset + filter_focus_offset + focus_wave_expect,
                )

                self.assertAlmostEqual(focusOffsetSummary.userApplied, 0.0)
                self.assertAlmostEqual(focusOffsetSummary.filter, filter_focus_offset)
                self.assertAlmostEqual(
                    focusOffsetSummary.disperser, disperser_focus_offset
                )

                offset = {"x": 1.3, "y": 1.4, "z": 1.5, "u": 1.6, "v": 1.7}

                # Now start the loops, we'll then add an offset,
                # then remove a filter, then remove a disperser
                # wavelength offsets should be removed when the filter changes
                logger.debug("Re-enabling atspectrograph and hexapod corrections")
                await harness.aos_remote.cmd_enableCorrection.set_start(
                    atspectrograph=True, hexapod=True
                )

                # add the userApplied-offsets
                await harness.aos_remote.cmd_offset.set_start(
                    **offset, timeout=STD_TIMEOUT
                )

                # flush events then send relative offsets
                harness.aos_remote.evt_correctionOffsets.flush()
                harness.aos_remote.evt_focusOffsetSummary.flush()

                logger.debug("Add a manual focus offset in z")
                user_focus_offset = 0.22
                await harness.aos_remote.cmd_applyFocusOffset.set_start(
                    offset=user_focus_offset, timeout=STD_TIMEOUT
                )

                # Command issues event before returning, so events should
                # be immediately available
                offset_applied = await harness.aos_remote.evt_correctionOffsets.aget(
                    timeout=STD_TIMEOUT
                )
                focusOffsetSummary = (
                    await harness.aos_remote.evt_focusOffsetSummary.aget(
                        timeout=STD_TIMEOUT
                    )
                )

                # offsets should be combined in z
                for axis in offset:
                    logger.debug(
                        "axis = {} and correction_offset"
                        " = {}".format(axis, getattr(offset_applied, axis))
                    )
                    with self.subTest(axis=axis):
                        if axis != "z":
                            self.assertAlmostEqual(
                                offset[axis], getattr(offset_applied, axis)
                            )
                        else:
                            # should be applied offset plus the
                            # filter/disperser offsets
                            self.assertAlmostEqual(
                                offset[axis]
                                + disperser_focus_offset
                                + filter_focus_offset
                                + focus_wave_expect
                                + user_focus_offset,
                                getattr(offset_applied, axis),
                            )

                # check that summary is correct
                self.assertAlmostEqual(
                    focusOffsetSummary.total, getattr(offset_applied, "z")
                )
                # userApplied offset should just be whatever we supplied
                self.assertAlmostEqual(
                    focusOffsetSummary.userApplied, offset["z"] + user_focus_offset
                )
                self.assertAlmostEqual(focusOffsetSummary.filter, filter_focus_offset)
                self.assertAlmostEqual(
                    focusOffsetSummary.disperser, disperser_focus_offset
                )
                self.assertAlmostEqual(focusOffsetSummary.wavelength, focus_wave_expect)

                logger.debug(
                    "Remove manual focus offset in z to ease accounting challenges"
                )
                await harness.aos_remote.cmd_applyFocusOffset.set_start(
                    offset=-user_focus_offset, timeout=STD_TIMEOUT
                )

                logger.debug("Putting in filter2")
                # flush events then change filters
                # wavelength offset should now go to zero
                focus_wave_expect = 0.0
                harness.aos_remote.evt_correctionOffsets.flush()
                harness.aos_remote.evt_focusOffsetSummary.flush()
                harness.aos_remote.evt_pointingOffsetSummary.flush()
                harness.aos_remote.evt_atspectrographCorrectionStarted.flush()
                harness.aos_remote.evt_atspectrographCorrectionCompleted.flush()

                harness.atspectrograph.evt_reportedFilterPosition.set_put(
                    name=filter_name2,
                    centralWavelength=filter_central_wavelength2,
                    focusOffset=filter_focus_offset2,
                    pointingOffsets=filter_pointing_offsets2,
                )

                # Timeouts extended as filter/disperser changes can take ~5s
                await harness.aos_remote.evt_atspectrographCorrectionStarted.next(
                    timeout=STD_TIMEOUT * 2, flush=False
                )
                await harness.aos_remote.evt_atspectrographCorrectionCompleted.next(
                    timeout=STD_TIMEOUT, flush=False
                )
                # check pointing offset was applied
                harness.atptg.cmd_poriginOffset.callback.assert_called()
                # Should have only been called once
                offset_call_count_filt2 = (
                    harness.atptg.cmd_poriginOffset.callback.call_count
                )
                # FIXME
                logger.debug(f"offset_call_count_filt2 is {offset_call_count_filt2}")

                # check focus model updates were applied
                offset_applied = await harness.aos_remote.evt_correctionOffsets.next(
                    flush=False, timeout=STD_TIMEOUT
                )
                focusOffsetSummary = (
                    await harness.aos_remote.evt_focusOffsetSummary.next(
                        flush=False, timeout=STD_TIMEOUT
                    )
                )

                # offsets should be combined in z
                # this could be made a function, but I found this easier
                # to read/parse/understand
                for axis in offset:

                    with self.subTest(axis=axis):
                        if axis != "z":
                            self.assertAlmostEqual(
                                offset[axis], getattr(offset_applied, axis)
                            )
                        else:
                            # should be applied offset plus the
                            # filter/disperser offsets
                            self.assertAlmostEqual(
                                offset[axis]
                                + disperser_focus_offset
                                + filter_focus_offset2,
                                getattr(offset_applied, axis),
                            )

                # check that focus summary is correct
                self.assertAlmostEqual(
                    focusOffsetSummary.total, getattr(offset_applied, "z")
                )
                # userApplied offset should just be whatever we supplied
                self.assertAlmostEqual(focusOffsetSummary.userApplied, offset["z"])
                self.assertAlmostEqual(focusOffsetSummary.filter, filter_focus_offset2)
                self.assertAlmostEqual(
                    focusOffsetSummary.disperser, disperser_focus_offset
                )
                # wavelength offset should now be zero!
                self.assertAlmostEqual(focusOffsetSummary.wavelength, focus_wave_expect)

                await asyncio.sleep(3)

                # flush events then change dispersers
                logger.debug("Putting in disperser2")
                harness.aos_remote.evt_correctionOffsets.flush()
                harness.aos_remote.evt_focusOffsetSummary.flush()
                harness.aos_remote.evt_pointingOffsetSummary.flush()
                harness.aos_remote.evt_atspectrographCorrectionStarted.flush()
                harness.aos_remote.evt_atspectrographCorrectionCompleted.flush()

                harness.atspectrograph.evt_reportedDisperserPosition.set_put(
                    name=disperser_name2,
                    focusOffset=disperser_focus_offset2,
                    pointingOffsets=disperser_pointing_offsets2,
                )
                # extended timeouts as filter/disperser changes takes ~5s
                await harness.aos_remote.evt_atspectrographCorrectionStarted.next(
                    timeout=STD_TIMEOUT * 2, flush=False
                )
                await harness.aos_remote.evt_atspectrographCorrectionCompleted.next(
                    timeout=STD_TIMEOUT, flush=False
                )
                # check pointing offset was applied
                offset_call_count_disp2 = (
                    harness.atptg.cmd_poriginOffset.callback.call_count
                )
                logger.debug(
                    f"offset_call_count_disp2 is now {offset_call_count_disp2}"
                )
                self.assertGreater(offset_call_count_disp2, offset_call_count_filt2)

                offset_applied = await harness.aos_remote.evt_correctionOffsets.next(
                    flush=False, timeout=STD_TIMEOUT * 2
                )
                focusOffsetSummary = (
                    await harness.aos_remote.evt_focusOffsetSummary.next(
                        flush=False, timeout=STD_TIMEOUT * 2
                    )
                )

                # offsets should be combined in z
                for axis in offset:

                    with self.subTest(axis=axis):
                        if axis != "z":
                            self.assertAlmostEqual(
                                offset[axis], getattr(offset_applied, axis)
                            )
                        else:
                            # should be applied offset plus the
                            # filter/disperser offsets
                            self.assertAlmostEqual(
                                offset[axis]
                                + disperser_focus_offset2
                                + filter_focus_offset2
                                + focus_wave_expect,
                                getattr(offset_applied, axis),
                            )

                # check that summary is correct
                self.assertAlmostEqual(
                    focusOffsetSummary.total, getattr(offset_applied, "z")
                )
                # userApplied offset should just be whatever we supplied
                self.assertAlmostEqual(focusOffsetSummary.userApplied, offset["z"])
                self.assertAlmostEqual(focusOffsetSummary.filter, filter_focus_offset2)
                self.assertAlmostEqual(
                    focusOffsetSummary.disperser, disperser_focus_offset2
                )
                # should be zero!
                self.assertAlmostEqual(focusOffsetSummary.wavelength, focus_wave_expect)

                # Now put in a filter that results in zero changes required
                # to the telescope setup, but make sure events are all
                # re-published.
                logger.debug("Putting in filter2b, which is filter2 again")
                # flush events then change filters
                # wavelength offset should now go to zero
                harness.aos_remote.evt_correctionOffsets.flush()
                harness.aos_remote.evt_focusOffsetSummary.flush()
                harness.aos_remote.evt_pointingOffsetSummary.flush()
                harness.aos_remote.evt_atspectrographCorrectionStarted.flush()
                harness.aos_remote.evt_atspectrographCorrectionCompleted.flush()

                harness.atspectrograph.evt_reportedFilterPosition.set_put(
                    name=filter_name2,
                    centralWavelength=filter_central_wavelength2,
                    focusOffset=filter_focus_offset2,
                    pointingOffsets=filter_pointing_offsets2,
                    force_output=True,  # Must force since it's the same event
                )

                # Timeouts extended as filter/disperser changes can take ~5s
                await harness.aos_remote.evt_atspectrographCorrectionStarted.next(
                    timeout=STD_TIMEOUT * 2, flush=False
                )
                await harness.aos_remote.evt_atspectrographCorrectionCompleted.next(
                    timeout=STD_TIMEOUT, flush=False
                )
                # check pointing offset was *NOT* applied since no
                # changes were made
                offset_call_count_filt2b = (
                    harness.atptg.cmd_poriginOffset.callback.call_count
                )
                self.assertEqual(offset_call_count_filt2b, offset_call_count_disp2)

                # check focus model updates were applied
                offset_applied2 = await harness.aos_remote.evt_correctionOffsets.next(
                    flush=False, timeout=STD_TIMEOUT
                )
                focusOffsetSummary2 = (
                    await harness.aos_remote.evt_focusOffsetSummary.next(
                        flush=False, timeout=STD_TIMEOUT
                    )
                )

                # check that summaries are unchanged
                self.assertAlmostEqual(
                    focusOffsetSummary.total, focusOffsetSummary2.total
                )
                self.assertAlmostEqual(offset_applied.z, offset_applied2.z)

                # Now reset the offsets (after flushing events), can only
                # reset offsets for hexapod loop as it's the only one enabled
                # will not reset spectrograph offsets!
                harness.aos_remote.evt_correctionOffsets.flush()
                harness.aos_remote.evt_focusOffsetSummary.flush()
                hexapod_axes = ["x", "y", "z", "u", "v"]
                for axis in hexapod_axes:
                    # Extend timeout a bit, 5s is sometimes too short now
                    # that we're not returning from commands right away
                    await harness.aos_remote.cmd_resetOffset.set_start(
                        axis=axis, timeout=STD_TIMEOUT * 2
                    )

                offset_applied = await harness.aos_remote.evt_correctionOffsets.aget(
                    timeout=STD_TIMEOUT
                )
                focusOffsetSummary = (
                    await harness.aos_remote.evt_focusOffsetSummary.aget(
                        timeout=STD_TIMEOUT
                    )
                )

                for axis in offset:
                    with self.subTest(axis=axis):
                        if axis != "z":
                            self.assertAlmostEqual(0.0, getattr(offset_applied, axis))
                        else:
                            # correction offset should be zero plus the
                            # filter/disperser offsets
                            self.assertAlmostEqual(
                                0.0 + disperser_focus_offset2 + filter_focus_offset2,
                                getattr(offset_applied, axis),
                            )
                # totals should be just filter/disperser offsets
                self.assertAlmostEqual(
                    focusOffsetSummary.total,
                    disperser_focus_offset2 + filter_focus_offset2,
                )
                # userApplied offset should be zero
                self.assertAlmostEqual(focusOffsetSummary.userApplied, 0.0)
                self.assertAlmostEqual(focusOffsetSummary.filter, filter_focus_offset2)
                self.assertAlmostEqual(
                    focusOffsetSummary.disperser, disperser_focus_offset2
                )

                # Disable corrections gracefully, makes debugging easier
                logger.debug("Disabling corrections")
                await harness.aos_remote.cmd_disableCorrection.set_start(
                    hexapod=True, atspectrograph=True, timeout=STD_TIMEOUT
                )

                # send spectrograph offline
                harness.atspectrograph.evt_summaryState.set_put(
                    summaryState=salobj.State.OFFLINE
                )

                # Bring to standby then enabled twice to verify bugfix
                # as part of DM-27243
                for i in range(2):
                    logger.debug(f"On iteration {i}, now enabling")
                    await salobj.set_summary_state(
                        harness.aos_remote, salobj.State.ENABLED, timeout=60
                    )
                    await asyncio.sleep(1)
                    logger.debug(f"On iteration {i}, now going to standby")
                    await salobj.set_summary_state(
                        harness.aos_remote, salobj.State.STANDBY, timeout=60
                    )

        asyncio.get_event_loop().run_until_complete(doit())

    def test_enable_disable_corrections(self):
        """Test enabling and disabling corrections, one at a time,
        then all at once."""

        async def doit(telescope_online=False):

            async with Harness() as harness:

                def callback(data):
                    pass

                def m1_open_callback(data):
                    harness.pnematics.evt_m1State.set_put(
                        state=ATPneumatics.AirValveState.OPENED
                    )

                def m2_open_callback(data):
                    harness.pnematics.evt_m2State.set_put(
                        state=ATPneumatics.AirValveState.OPENED
                    )

                def m1_close_callback(data):
                    harness.pnematics.evt_m1State.set_put(
                        state=ATPneumatics.AirValveState.CLOSED
                    )

                def m2_close_callback(data):
                    harness.pnematics.evt_m2State.set_put(
                        state=ATPneumatics.AirValveState.CLOSED
                    )

                # set the hexapod callback
                def hexapod_move_callback(data):
                    harness.hexapod.evt_positionUpdate.put()

                async def mount_offset_callback(self):
                    # just assume 0.1 degree offets
                    harness.atmcs.evt_allAxesInPosition.set_put(inPosition=False)
                    await publish_mountEncoders(
                        harness, azimuth + 0.1, elevation + 0.1, ntimes=1
                    )
                    harness.atmcs.evt_allAxesInPosition.set_put(inPosition=True)

                harness.pnematics.cmd_m1SetPressure.callback = Mock(wraps=callback)
                harness.pnematics.cmd_m2SetPressure.callback = Mock(wraps=callback)
                harness.pnematics.cmd_m1OpenAirValve.callback = Mock(
                    wraps=m1_open_callback
                )
                harness.pnematics.cmd_m2OpenAirValve.callback = Mock(
                    wraps=m2_open_callback
                )
                harness.pnematics.cmd_m1CloseAirValve.callback = Mock(
                    wraps=m1_close_callback
                )
                harness.pnematics.cmd_m2CloseAirValve.callback = Mock(
                    wraps=m2_close_callback
                )
                harness.hexapod.cmd_moveToPosition.callback = Mock(
                    wraps=hexapod_move_callback
                )
                harness.atptg.cmd_poriginOffset.callback = Mock(
                    wraps=mount_offset_callback
                )

                harness.pnematics.evt_summaryState.set_put(
                    summaryState=salobj.State.ENABLED
                )
                harness.pnematics.evt_mainValveState.set_put(
                    state=ATPneumatics.AirValveState.OPENED
                )
                harness.pnematics.evt_instrumentState.set_put(
                    state=ATPneumatics.AirValveState.OPENED
                )
                harness.pnematics.evt_m1State.set_put(
                    state=ATPneumatics.AirValveState.CLOSED
                )
                harness.pnematics.evt_m2State.set_put(
                    state=ATPneumatics.AirValveState.CLOSED
                )

                # report atspectrograph status
                harness.atspectrograph.evt_summaryState.set_put(
                    summaryState=salobj.State.ENABLED
                )
                filterFocusOffset, disperserFocusOffset = 0.1, 0.05
                filterPointingOffsets = np.array([0.03, -0.03])
                gratingPointingOffsets = np.array([0.1, -0.1])

                harness.atspectrograph.evt_reportedFilterPosition.set_put(
                    name="filter1",
                    centralWavelength=701.5,
                    focusOffset=filterFocusOffset,
                    pointingOffsets=filterPointingOffsets,
                )
                harness.atspectrograph.evt_reportedDisperserPosition.set_put(
                    name="disperser2",
                    focusOffset=disperserFocusOffset,
                    pointingOffsets=gratingPointingOffsets,
                )

                # Put ataos in enabled state
                # extend the timeout, to account for if all events aren't
                # set yet
                await asyncio.sleep(1)

                await salobj.set_summary_state(
                    harness.aos_remote, salobj.State.ENABLED, timeout=80
                )
                self.assertEqual(harness.csc.summary_state, salobj.State.ENABLED)

                if telescope_online:
                    azimuth, elevation = 25.0, 60.0
                    await publish_mountEncoders(harness, azimuth, elevation, ntimes=2)
                else:
                    azimuth, elevation = None, None

                # Try to send empty enable correction, this will fail.
                logger.debug("Try to send empty enable correction, this will fail.")
                with self.assertRaises(salobj.AckError):
                    await harness.aos_remote.cmd_enableCorrection.set_start(
                        timeout=STD_TIMEOUT
                    )

                # Enable corrections one by one
                corrections = (
                    "m1",
                    "m2",
                    "hexapod",
                    "focus",
                    "moveWhileExposing",
                    "atspectrograph",
                )
                expected_corrections = {
                    "m1": False,
                    "m2": False,
                    "hexapod": False,
                    "focus": False,
                    "moveWhileExposing": False,
                    "atspectrograph": False,
                }

                logger.debug("Starting to loop over enabling corrections")
                for corr in corrections:
                    expected_corrections[corr] = True
                    # Note: Setting only the correction I want to enable.
                    # Any correction already enabled will be left unchanged.
                    # This is part of the test.
                    logger.debug(f"Enabling correction {corr}")

                    # flush correctionEnabled event
                    harness.aos_remote.evt_correctionEnabled.flush()
                    harness.aos_remote.evt_hexapodCorrectionCompleted.flush()
                    harness.aos_remote.evt_atspectrographCorrectionStarted.flush()
                    harness.aos_remote.evt_atspectrographCorrectionCompleted.flush()
                    harness.aos_remote.evt_focusOffsetSummary.flush()
                    harness.aos_remote.evt_pointingOffsetSummary.flush()
                    # send command to start
                    await harness.aos_remote.cmd_enableCorrection.set_start(
                        **expected_corrections
                    )

                    # grab correction enabled event
                    correctionEnabledEvent = (
                        await harness.aos_remote.evt_correctionEnabled.next(
                            flush=False, timeout=STD_TIMEOUT
                        )
                    )

                    # let the loop turn a few times to make sure it's stable
                    # crash
                    logger.debug(f"Correction {corr} enabled")

                    # This is inefficient but leaving it for ease of
                    # readability
                    if telescope_online:
                        if corr == "m1":
                            await harness.aos_remote.evt_m1CorrectionStarted.aget(
                                timeout=STD_TIMEOUT
                            )
                            harness.pnematics.cmd_m1SetPressure.callback.assert_called()
                            harness.pnematics.cmd_m1OpenAirValve.callback.assert_called()
                            await harness.aos_remote.evt_m1CorrectionCompleted.aget(
                                timeout=STD_TIMEOUT
                            )
                        elif corr == "m2":
                            await harness.aos_remote.evt_m2CorrectionStarted.aget(
                                timeout=STD_TIMEOUT
                            )
                            harness.pnematics.cmd_m2SetPressure.callback.assert_called()
                            harness.pnematics.cmd_m2OpenAirValve.callback.assert_called()
                            await harness.aos_remote.evt_m2CorrectionCompleted.aget(
                                timeout=STD_TIMEOUT
                            )
                        elif corr == "hexapod":
                            await harness.aos_remote.evt_hexapodCorrectionStarted.next(
                                timeout=STD_TIMEOUT, flush=False
                            )
                            harness.hexapod.cmd_moveToPosition.callback.assert_called()
                            await harness.aos_remote.evt_hexapodCorrectionCompleted.next(
                                timeout=STD_TIMEOUT, flush=False
                            )
                        elif corr == "atspectrograph":
                            await harness.aos_remote.evt_atspectrographCorrectionStarted.next(
                                timeout=STD_TIMEOUT, flush=False
                            )
                            await harness.aos_remote.evt_atspectrographCorrectionCompleted.next(
                                timeout=STD_TIMEOUT, flush=False
                            )
                            # check pointing offset was applied
                            harness.atptg.cmd_poriginOffset.callback.assert_called()
                            # check offsets were applied correctly
                            focusOffsetSummary = (
                                await harness.aos_remote.evt_focusOffsetSummary.next(
                                    flush=False, timeout=STD_TIMEOUT
                                )
                            )
                            self.assertAlmostEqual(
                                focusOffsetSummary.total,
                                disperserFocusOffset + filterFocusOffset,
                            )
                            # userApplied offset should be zero
                            self.assertAlmostEqual(focusOffsetSummary.userApplied, 0.0)
                            self.assertAlmostEqual(
                                focusOffsetSummary.filter, filterFocusOffset
                            )
                            self.assertAlmostEqual(
                                focusOffsetSummary.disperser, disperserFocusOffset
                            )
                            pointingOffsetSummary = (
                                await harness.aos_remote.evt_pointingOffsetSummary.next(
                                    flush=False, timeout=STD_TIMEOUT
                                )
                            )
                            # pointingOffsets are arrays, so loop over values
                            # individually
                            for n in range(len(pointingOffsetSummary.total) - 1):
                                self.assertAlmostEqual(
                                    pointingOffsetSummary.filter[n],
                                    filterPointingOffsets[n],
                                )
                                self.assertAlmostEqual(
                                    pointingOffsetSummary.disperser[n],
                                    gratingPointingOffsets[n],
                                )
                                self.assertAlmostEqual(
                                    pointingOffsetSummary.total[n],
                                    filterPointingOffsets[n]
                                    + gratingPointingOffsets[n],
                                )
                    else:
                        # when the telescope is offline, the m1 and m2 loops
                        # can still run and should not raise an exception
                        if corr == "m1":
                            harness.pnematics.cmd_m1SetPressure.callback.assert_called()
                            harness.pnematics.cmd_m1OpenAirValve.callback.assert_called()
                        elif corr == "m2":
                            harness.pnematics.cmd_m2SetPressure.callback.assert_called()
                            harness.pnematics.cmd_m2OpenAirValve.callback.assert_called()

                    # check all corrections are enabled
                    for test_corr in corrections:
                        with self.subTest(test_corr=test_corr):
                            self.assertEqual(
                                getattr(correctionEnabledEvent, test_corr),
                                expected_corrections[test_corr],
                                f"Failed to set correction for {test_corr}. "
                                f"Expected {expected_corrections[test_corr]}, "
                                f"got {getattr(correctionEnabledEvent, test_corr)}",
                            )

                logger.debug(
                    f"All corrections should be enabled \n {correctionEnabledEvent}"
                )

                await asyncio.sleep(0.0)
                # Try to send empty disable correction, this will fail.
                logger.debug("Try to send empty disable correction, this should fail")
                with self.assertRaises(salobj.AckError):
                    await harness.aos_remote.cmd_disableCorrection.set_start()

                # Disable corrections one by one
                for corr in corrections:

                    # mark as expected to False
                    expected_corrections[corr] = False
                    # Note: Setting only the correction I want to disable.
                    # Any correction already disabled will be left unchanged.
                    # This is part of the test.

                    # True means I want to disable
                    specific_correction_to_disable = {corr: True}

                    harness.aos_remote.evt_correctionEnabled.flush()
                    await harness.aos_remote.cmd_disableCorrection.set_start(
                        **specific_correction_to_disable
                    )
                    correctionEnabledEvent = (
                        await harness.aos_remote.evt_correctionEnabled.next(
                            flush=False, timeout=STD_TIMEOUT
                        )
                    )

                    for test_corr in corrections:
                        with self.subTest(test_corr=test_corr):
                            self.assertEqual(
                                getattr(correctionEnabledEvent, test_corr),
                                expected_corrections[test_corr],
                                f"Failed to set correction for {test_corr}. "
                                f"Expected {expected_corrections[test_corr]}, "
                                f"got {getattr(correctionEnabledEvent, test_corr)}",
                            )

                await asyncio.sleep(0.0)
                logger.debug(
                    f"All corrections should be disabled \n {correctionEnabledEvent}"
                )

                logger.debug("Everything is disable, send enable all.")
                harness.aos_remote.evt_correctionEnabled.flush()
                await harness.aos_remote.cmd_enableCorrection.set_start(
                    enableAll=True, moveWhileExposing=False
                )
                correctionEnabledEvent = (
                    await harness.aos_remote.evt_correctionEnabled.next(
                        flush=False, timeout=STD_TIMEOUT
                    )
                )

                # All corrections should all be true, except moveWhileExposing
                for test_corr in corrections:
                    logger.debug(f"test_corr is {test_corr}")
                    with self.subTest(test_corr=test_corr):
                        if test_corr == "moveWhileExposing":
                            self.assertFalse(
                                getattr(correctionEnabledEvent, test_corr),
                                "Failed to set correction for moveWhileExposing. "
                                "Expected True, "
                                f"got {getattr(correctionEnabledEvent, test_corr)}",
                            )
                        else:
                            self.assertTrue(
                                getattr(correctionEnabledEvent, test_corr),
                                "Failed to set correction for {test_corr}. "
                                "Expected True, "
                                f"got {getattr(correctionEnabledEvent, test_corr)}",
                            )

                logger.debug(
                    "All corrections should be enabled except moveWhileExposing:\n"
                    f" {correctionEnabledEvent}"
                )

                logger.debug(
                    "Everything should be enabled except moveWhileExposing, "
                    "now sending disable all."
                )
                harness.aos_remote.evt_correctionEnabled.flush()
                await harness.aos_remote.cmd_disableCorrection.set_start(
                    disableAll=True
                )
                correctionEnabledEvent = (
                    await harness.aos_remote.evt_correctionEnabled.next(
                        flush=False, timeout=STD_TIMEOUT
                    )
                )

                # They should all be False
                for test_corr in corrections:
                    with self.subTest(test_corr=test_corr):
                        self.assertFalse(
                            getattr(correctionEnabledEvent, test_corr),
                            "Failed to set correction for {test_corr}. "
                            "Expected False, "
                            f"got {getattr(correctionEnabledEvent, test_corr)}",
                        )

                logger.debug(
                    f"All corrections should be disabled except moveWhileExposing:\n{correctionEnabledEvent}"
                )

                # send to disabled state so loop errors don't plague output
                logger.debug(
                    "Setting ATAOS to disabled state at end of test_enable_disable_corrections"
                )
                await salobj.set_summary_state(
                    harness.aos_remote, salobj.State.DISABLED, timeout=80
                )

        logger.debug(
            "\n Starting test_enable_disable_corrections without telescope position published"
        )
        asyncio.get_event_loop().run_until_complete(doit(telescope_online=False))
        logger.debug(
            "\n test_enable_disable_corrections without telescope position - COMPLETED \n"
        )

        logger.debug(
            "\n Starting test_enable_disable_corrections with telescope position published"
        )
        asyncio.get_event_loop().run_until_complete(doit(telescope_online=True))

    def test_target_handling(self):
        """Test changing of targets to verify pressures are adjusted
        correctly.
        """

        async def doit():
            async with Harness() as harness:

                def callback(data):
                    pass

                def m1_open_callback(data):
                    harness.pnematics.evt_m1State.set_put(
                        state=ATPneumatics.AirValveState.OPENED
                    )

                def m2_open_callback(data):
                    harness.pnematics.evt_m2State.set_put(
                        state=ATPneumatics.AirValveState.OPENED
                    )

                def m1_close_callback(data):
                    harness.pnematics.evt_m1State.set_put(
                        state=ATPneumatics.AirValveState.CLOSED
                    )

                def m2_close_callback(data):
                    harness.pnematics.evt_m2State.set_put(
                        state=ATPneumatics.AirValveState.CLOSED
                    )

                # set the hexapod callback
                def hexapod_move_callback(data):
                    harness.hexapod.evt_positionUpdate.put()

                harness.pnematics.cmd_m1SetPressure.callback = Mock(wraps=callback)
                harness.pnematics.cmd_m2SetPressure.callback = Mock(wraps=callback)
                harness.pnematics.cmd_m1OpenAirValve.callback = Mock(
                    wraps=m1_open_callback
                )
                harness.pnematics.cmd_m2OpenAirValve.callback = Mock(
                    wraps=m2_open_callback
                )
                harness.pnematics.cmd_m1CloseAirValve.callback = Mock(
                    wraps=m1_close_callback
                )
                harness.pnematics.cmd_m2CloseAirValve.callback = Mock(
                    wraps=m2_close_callback
                )
                harness.hexapod.cmd_moveToPosition.callback = Mock(
                    wraps=hexapod_move_callback
                )

                harness.pnematics.evt_summaryState.set_put(
                    summaryState=salobj.State.ENABLED
                )
                harness.pnematics.evt_mainValveState.set_put(
                    state=ATPneumatics.AirValveState.OPENED
                )
                harness.pnematics.evt_instrumentState.set_put(
                    state=ATPneumatics.AirValveState.OPENED
                )
                harness.pnematics.evt_m1State.set_put(
                    state=ATPneumatics.AirValveState.CLOSED
                )
                harness.pnematics.evt_m2State.set_put(
                    state=ATPneumatics.AirValveState.CLOSED
                )

                # report atspectrograph status
                harness.atspectrograph.evt_summaryState.set_put(
                    summaryState=salobj.State.ENABLED
                )
                filterFocusOffset, disperserFocusOffset = 0.1, 0.05
                # Set offsets to zero to make assertions easier
                filterPointingOffsets = np.array([0.0, 0.00])
                gratingPointingOffsets = np.array([0.0, 0.0])

                harness.atspectrograph.evt_reportedFilterPosition.set_put(
                    name="filter1",
                    centralWavelength=701.5,
                    focusOffset=filterFocusOffset,
                    pointingOffsets=filterPointingOffsets,
                )
                harness.atspectrograph.evt_reportedDisperserPosition.set_put(
                    name="disperser2",
                    focusOffset=disperserFocusOffset,
                    pointingOffsets=gratingPointingOffsets,
                )

                await salobj.set_summary_state(
                    harness.aos_remote,
                    salobj.State.ENABLED,
                    settingsToApply="current",
                    timeout=90,
                )
                self.assertEqual(harness.csc.summary_state, salobj.State.ENABLED)

                # Verify the elevation and azimuth positions and
                # targets are at none
                self.assertEqual(harness.csc.elevation, None)
                self.assertEqual(harness.csc.target_elevation, None)
                self.assertEqual(harness.csc.azimuth, None)
                self.assertEqual(harness.csc.target_azimuth, None)

                logger.debug("Enable all corrections")
                await harness.aos_remote.cmd_enableCorrection.set_start(
                    enableAll=True, moveWhileExposing=False
                )

                logger.debug("Send a target")
                t_el = 80.0
                t_az = 70.0
                t_nas2 = t_az
                harness.atmcs.evt_target.set_put(
                    elevation=t_el, azimuth=t_az, nasmyth2RotatorAngle=t_nas2
                )
                # wait for loop to catch it
                await asyncio.sleep(2)
                # Verify the elevation and azimuth positions and targets
                # are at none
                self.assertEqual(harness.csc.elevation, None)
                self.assertEqual(harness.csc.target_elevation, t_el)
                self.assertEqual(harness.csc.azimuth, None)
                self.assertEqual(harness.csc.target_azimuth, t_az)

                logger.debug("Send a telescope position")
                await publish_mountEncoders(harness, t_az, t_el, ntimes=2)
                self.assertEqual(harness.csc.elevation, t_el)
                self.assertEqual(harness.csc.target_elevation, t_el)
                self.assertEqual(harness.csc.azimuth, t_az)
                self.assertEqual(harness.csc.target_azimuth, t_az)

                logger.debug("Send a lower telescope position")
                # Want to check that pressure is decreased faster than the
                # model value for the current position so the mirror
                # doesn't lift
                t_el2 = 40.0
                harness.atmcs.evt_target.set_put(
                    elevation=t_el2, azimuth=t_az, nasmyth2RotatorAngle=t_nas2
                )
                # wait for loop to catch it
                await asyncio.sleep(2)
                self.assertEqual(harness.csc.elevation, t_el)
                self.assertEqual(harness.csc.target_elevation, t_el2)
                self.assertEqual(harness.csc.azimuth, t_az)
                self.assertEqual(harness.csc.target_azimuth, t_az)

                # Check that m1 pressure is not equal to current position
                m1_pressure_expected_to_be_commanded = (
                    harness.csc.model.get_correction_m1(t_az, (t_el + t_el2) / 2.0)
                )

                # returns a call object, so need to take the first call,
                # then the value of the tuple, then the pressure value
                commanded_pressure = (
                    ((harness.pnematics.cmd_m1SetPressure.callback.call_args)[0])[0]
                ).pressure
                self.assertEqual(
                    commanded_pressure, m1_pressure_expected_to_be_commanded
                )

                logger.debug("Update telescope position to intermediate position")
                t_el_1b = 45.0
                await publish_mountEncoders(harness, t_az, t_el_1b, ntimes=2)
                self.assertEqual(harness.csc.elevation, t_el_1b)
                self.assertEqual(harness.csc.target_elevation, t_el2)
                self.assertEqual(harness.csc.azimuth, t_az)
                self.assertEqual(harness.csc.target_azimuth, t_az)

                # Check that m1 pressure is not equal to current position
                m1_pressure_expected_to_be_commanded = (
                    harness.csc.model.get_correction_m1(t_az, (t_el_1b + t_el2) / 2.0)
                )
                commanded_pressure = (
                    ((harness.pnematics.cmd_m1SetPressure.callback.call_args)[0])[0]
                ).pressure
                self.assertEqual(
                    commanded_pressure, m1_pressure_expected_to_be_commanded
                )

                logger.debug("Update telescope position to be target position")
                await publish_mountEncoders(harness, t_az, t_el2, ntimes=2)
                self.assertEqual(harness.csc.elevation, t_el2)
                self.assertEqual(harness.csc.target_elevation, t_el2)
                self.assertEqual(harness.csc.azimuth, t_az)
                self.assertEqual(harness.csc.target_azimuth, t_az)

                # Check that m1 pressure is not equal to current position
                m1_pressure_expected_to_be_commanded = (
                    harness.csc.model.get_correction_m1(t_az, t_el2)
                )
                commanded_pressure = (
                    ((harness.pnematics.cmd_m1SetPressure.callback.call_args)[0])[0]
                ).pressure
                self.assertEqual(
                    commanded_pressure, m1_pressure_expected_to_be_commanded
                )

                logger.debug("Send target for original position")
                harness.atmcs.evt_target.set_put(
                    elevation=t_el, azimuth=t_az, nasmyth2RotatorAngle=t_nas2
                )

                logger.debug("Update telescope position to intermediate position")
                t_el_1b = 45.0
                await publish_mountEncoders(harness, t_az, t_el_1b, ntimes=2)

                self.assertEqual(harness.csc.elevation, t_el_1b)
                self.assertEqual(harness.csc.target_elevation, t_el)
                self.assertEqual(harness.csc.azimuth, t_az)
                self.assertEqual(harness.csc.target_azimuth, t_az)

                # Check that m1 pressure is not equal to current position
                # On the way up the air pressure follows telescope position
                m1_pressure_expected_to_be_commanded = (
                    harness.csc.model.get_correction_m1(t_az, t_el_1b)
                )
                commanded_pressure = (
                    ((harness.pnematics.cmd_m1SetPressure.callback.call_args)[0])[0]
                ).pressure
                self.assertEqual(
                    commanded_pressure, m1_pressure_expected_to_be_commanded
                )

                logger.debug("Switch corrections off")
                harness.aos_remote.cmd_disableCorrection.set(disableAll=True)

        logger.debug("\n Starting test_target_handling")
        asyncio.get_event_loop().run_until_complete(doit())
        logger.debug("\n test_target_handling - COMPLETED \n")


async def publish_mountEncoders(harness, azimuth, elevation, ntimes=5):
    """Publish telescope position as an event.
    Nasmyth values are just equal to azimuth
    """

    # arrays need to have a length of 100 values
    for i in range(ntimes):
        _azimuth = np.zeros(100) + azimuth
        # make sure it is never zero because np.random.uniform is [min, max)
        _elevation = np.zeros(100) + elevation
        harness.atmcs.tel_mount_AzEl_Encoders.set_put(
            azimuthCalculatedAngle=_azimuth, elevationCalculatedAngle=_elevation
        )
        # Assume nasmyth is the same as the azimuth
        harness.atmcs.tel_mount_Nasmyth_Encoders.set_put(
            nasmyth1CalculatedAngle=_azimuth,
            nasmyth2CalculatedAngle=_azimuth,
        )
        await asyncio.sleep(salobj.base_csc.HEARTBEAT_INTERVAL)


if __name__ == "__main__":
    unittest.main()
