import sys
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
logger.level = logging.DEBUG

STD_TIMEOUT = 5  # standard command timeout (sec)
LONG_TIMEOUT = 20  # timeout for starting SAL components (sec)
TEST_CONFIG_DIR = pathlib.Path(__file__).parents[1].joinpath("tests", "data", "config")


class Harness:
    def __init__(self, config_dir=None):
        salobj.test_utils.set_random_lsst_dds_domain()
        self.csc = ataos_csc.ATAOS(config_dir=config_dir)

        # Adds a remote to control the ATAOS CSC
        self.aos_remote = salobj.Remote(self.csc.domain, "ATAOS")

        # Adds Controllers to receive commands from the ATAOS system
        self.atmcs = salobj.Controller("ATMCS")
        self.pnematics = salobj.Controller("ATPneumatics")
        self.hexapod = salobj.Controller("ATHexapod")
        self.camera = salobj.Controller("ATCamera")
        self.atspectrograph = salobj.Controller("ATSpectrograph")

        # set the debug level to be whatever is set above. Note that this statement *MUST* occur after
        # the controllers are created
        self.csc.log.level = logger.level

        # set the command timeout to be small so we don't have to wait for errors
        self.csc.cmd_timeout = 5.

    async def __aenter__(self):
        await asyncio.gather(self.csc.start_task,
                             self.aos_remote.start_task,
                             self.atmcs.start_task,
                             self.pnematics.start_task,
                             self.hexapod.start_task,
                             self.camera.start_task,
                             self.atspectrograph.start_task)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await asyncio.gather(self.csc.close(),
                             self.aos_remote.close(),
                             self.atmcs.close(),
                             self.pnematics.close(),
                             self.hexapod.close(),
                             self.camera.close(),
                             self.atspectrograph.close())


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

            extra_commands = ("applyFocusOffset",
                              "disableCorrection",
                              "enableCorrection")

            async with Harness() as harness:

                # Check initial state
                current_state = await harness.aos_remote.evt_summaryState.next(flush=False,
                                                                               timeout=1.)

                self.assertEqual(harness.csc.summary_state, salobj.State.STANDBY)
                self.assertEqual(current_state.summaryState, salobj.State.STANDBY)

                # Check that settingVersions was published and matches expected values
                setting_versions = await harness.aos_remote.evt_settingVersions.next(flush=False,
                                                                                     timeout=1.)
                self.assertIsNotNone(setting_versions)

                for bad_command in commands:
                    if bad_command in ("start", "exitControl"):
                        continue  # valid command in STANDBY state
                    with self.subTest(bad_command=bad_command):
                        cmd_attr = getattr(harness.aos_remote, f"cmd_{bad_command}")
                        with self.assertRaises(salobj.AckError):
                            id_ack = await cmd_attr.start(cmd_attr.DataType(), timeout=1.)

                for bad_command in extra_commands:
                    with self.subTest(bad_command=bad_command):
                        cmd_attr = getattr(harness.aos_remote, f"cmd_{bad_command}")
                        with self.assertRaises(salobj.AckError):
                            id_ack = await cmd_attr.start(cmd_attr.DataType(), timeout=1.)

                # send start; new state is DISABLED
                cmd_attr = getattr(harness.aos_remote, f"cmd_start")
                harness.aos_remote.evt_summaryState.flush()
                id_ack = await cmd_attr.start(timeout=120)  # this one can take longer to execute
                state = await harness.aos_remote.evt_summaryState.next(flush=False, timeout=5.)
                self.assertEqual(id_ack.ack, salobj.SalRetCode.CMD_COMPLETE)
                self.assertEqual(id_ack.error, 0)
                self.assertEqual(harness.csc.summary_state, salobj.State.DISABLED)
                self.assertEqual(state.summaryState, salobj.State.DISABLED)

                # TODO: There are two events issued when starting; appliedSettingsMatchStart and
                # settingsApplied. Check that they are received.

                for bad_command in commands:
                    if bad_command in ("enable", "standby"):
                        continue  # valid command in DISABLED state
                    with self.subTest(bad_command=bad_command):
                        cmd_attr = getattr(harness.aos_remote, f"cmd_{bad_command}")
                        with self.assertRaises(salobj.AckError):
                            id_ack = await cmd_attr.start(cmd_attr.DataType(), timeout=1.)

                for bad_command in extra_commands:
                    with self.subTest(bad_command=bad_command):
                        cmd_attr = getattr(harness.aos_remote, f"cmd_{bad_command}")
                        with self.assertRaises(salobj.AckError):
                            id_ack = await cmd_attr.start(cmd_attr.DataType(), timeout=1.)

                # send enable; new state is ENABLED
                cmd_attr = getattr(harness.aos_remote, f"cmd_enable")
                harness.aos_remote.evt_summaryState.flush()
                id_ack = await cmd_attr.start(timeout=120)  # this one can take longer to execute
                state = await harness.aos_remote.evt_summaryState.next(flush=False, timeout=5.)
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
                            id_ack = await cmd_attr.start(cmd_attr.DataType(), timeout=1.)

                # Todo: Test that other commands works.
                # send disable; new state is DISABLED
                cmd_attr = getattr(harness.aos_remote, f"cmd_disable")
                # this CMD may take some time to complete
                id_ack = await cmd_attr.start(cmd_attr.DataType(), timeout=30.)
                self.assertEqual(id_ack.ack, salobj.SalRetCode.CMD_COMPLETE)
                self.assertEqual(id_ack.error, 0)
                self.assertEqual(harness.csc.summary_state, salobj.State.DISABLED)

        asyncio.get_event_loop().run_until_complete(doit())

    def test_applyCorrection(self):
        """Test applyCorrection command. This commands applies the corrections for the current
        telescope position. It only works when the correction loop is not enabled."""

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
                    harness.pnematics.evt_m1State.set_put(state=ATPneumatics.AirValveState.OPENED)

                def m2_open_callback(data):
                    harness.pnematics.evt_m2State.set_put(state=ATPneumatics.AirValveState.OPENED)

                def m1_close_callback(data):
                    harness.pnematics.evt_m1State.set_put(state=ATPneumatics.AirValveState.CLOSED)

                def m2_close_callback(data):
                    harness.pnematics.evt_m2State.set_put(state=ATPneumatics.AirValveState.CLOSED)

                harness.pnematics.cmd_m1SetPressure.callback = Mock(wraps=callback)
                harness.pnematics.cmd_m2SetPressure.callback = Mock(wraps=callback)
                harness.pnematics.cmd_m1OpenAirValve.callback = Mock(wraps=m1_open_callback)
                harness.pnematics.cmd_m2OpenAirValve.callback = Mock(wraps=m2_open_callback)
                harness.pnematics.cmd_m1CloseAirValve.callback = Mock(wraps=m1_close_callback)
                harness.pnematics.cmd_m2CloseAirValve.callback = Mock(wraps=m2_close_callback)

                harness.pnematics.evt_summaryState.set_put(summaryState=salobj.State.ENABLED)
                harness.pnematics.evt_mainValveState.set_put(state=ATPneumatics.AirValveState.OPENED)
                harness.pnematics.evt_instrumentState.set_put(state=ATPneumatics.AirValveState.OPENED)
                harness.pnematics.evt_m1State.set_put(state=ATPneumatics.AirValveState.CLOSED)
                harness.pnematics.evt_m2State.set_put(state=ATPneumatics.AirValveState.CLOSED)

                harness.atspectrograph.evt_summaryState.set_put(summaryState=salobj.State.ENABLED)

                # FIXME: Check if this is correct! Is there a difference in
                # command to move hexapod and focus or they will use the same
                # command?
                harness.hexapod.cmd_moveToPosition.callback = Mock(wraps=hexapod_move_callback)

                # Add callback to detailedState events
                harness.aos_remote.evt_detailedState.callback = Mock(wraps=callback)

                timeout = 40. * salobj.base_csc.HEARTBEAT_INTERVAL

                logger.debug('Enabling ataos')
                await salobj.set_summary_state(harness.aos_remote, salobj.State.ENABLED)
                self.assertEqual(harness.csc.summary_state, salobj.State.ENABLED)
                logger.debug('Enabled ataos')

                logger.debug('Check that applyCorrection fails if enable Correction is on')
                harness.aos_remote.cmd_enableCorrection.set(m1=True)
                await harness.aos_remote.cmd_enableCorrection.start(timeout=timeout)

                with self.assertRaises(salobj.AckError):
                    await harness.aos_remote.cmd_applyCorrection.start(timeout=timeout)

                logger.debug('Switch corrections off')
                harness.aos_remote.cmd_disableCorrection.set(disableAll=True)
                await harness.aos_remote.cmd_disableCorrection.start(timeout=timeout)

                coro_m1_start = harness.aos_remote.evt_m1CorrectionStarted.next(flush=False,
                                                                                timeout=timeout)
                coro_m2_start = harness.aos_remote.evt_m2CorrectionStarted.next(flush=False,
                                                                                timeout=timeout)
                coro_hx_start = harness.aos_remote.evt_hexapodCorrectionStarted.next(flush=False,
                                                                                     timeout=timeout)

                coro_m1_end = harness.aos_remote.evt_m1CorrectionCompleted.next(flush=False,
                                                                                timeout=timeout)
                coro_m2_end = harness.aos_remote.evt_m2CorrectionCompleted.next(flush=False,
                                                                                timeout=timeout)
                coro_hx_end = harness.aos_remote.evt_hexapodCorrectionCompleted.next(flush=False,
                                                                                     timeout=timeout)

                logger.debug('Send applyCorrection command')
                azimuth = np.random.uniform(0., 360.)
                # make sure it is never zero because np.random.uniform is [min, max)
                elevation = 90. - np.random.uniform(0., 90.)

                await publish_mountEncoders(harness, azimuth, elevation, ntimes=5)

                logger.debug("Test that the hexapod won't move if there's an exposure happening")
                if while_exposing:
                    shutter_state_topic = harness.camera.evt_shutterDetailedState.DataType()
                    shutter_state_topic.substate = ataos_csc.ShutterState.OPEN
                    harness.camera.evt_shutterDetailedState.put(shutter_state_topic)
                    # Give some time for the CSC to grab that event
                    await asyncio.sleep(2.)

                if not get_tel_pos:
                    await asyncio.sleep(5 * salobj.base_csc.HEARTBEAT_INTERVAL)
                    await harness.aos_remote.cmd_applyCorrection.set_start(azimuth=azimuth,
                                                                           elevation=elevation,
                                                                           timeout=timeout)
                else:
                    await asyncio.sleep(5 * salobj.base_csc.HEARTBEAT_INTERVAL)
                    await harness.aos_remote.cmd_applyCorrection.start(timeout=timeout)

                # Give control back to event loop so it can gather remaining callbacks
                await asyncio.sleep(5 * salobj.base_csc.HEARTBEAT_INTERVAL)

                logger.debug("Check that callbacks where called")
                harness.pnematics.cmd_m1SetPressure.callback.assert_called()
                harness.pnematics.cmd_m2SetPressure.callback.assert_called()
                if while_exposing:
                    harness.hexapod.cmd_moveToPosition.callback.assert_not_called()
                else:
                    harness.hexapod.cmd_moveToPosition.callback.assert_called()
                harness.aos_remote.evt_detailedState.callback.assert_called()

                # Check that events where published with the correct az/el position
                m1_start = await coro_m1_start
                m2_start = await coro_m2_start

                m1_end = await coro_m1_end
                m2_end = await coro_m2_end

                if while_exposing:
                    hx_start = None
                    hx_end = None
                    # hexapod should timeout
                    with self.assertRaises(asyncio.TimeoutError):
                        await coro_hx_start
                    with self.assertRaises(asyncio.TimeoutError):
                        await coro_hx_end
                else:
                    hx_start = await coro_hx_start
                    hx_end = await coro_hx_end

                for component in (m1_start, m2_start, hx_start, m1_end, m2_end, hx_end):
                    if component is None:
                        continue

                    with self.subTest(component=component, azimuth=azimuth):
                        self.assertEqual(component.azimuth, azimuth)
                    with self.subTest(component=component, elevation=elevation):
                        self.assertEqual(component.elevation, elevation)

                self.assertEqual(
                    len(harness.aos_remote.evt_detailedState.callback.call_args_list),
                    6 if not while_exposing else 4,
                    '%s' % harness.aos_remote.evt_detailedState.callback.call_args_list
                )

                logger.debug("Disable ATAOS CSC")
                await harness.aos_remote.cmd_disable.start(timeout=timeout)

        # Run test getting the telescope position
        asyncio.get_event_loop().run_until_complete(doit(get_tel_pos=True))
        logger.debug("test getting the telescope position - COMPLETE")

        # Run test getting the telescope position while exposing
        asyncio.get_event_loop().run_until_complete(doit(get_tel_pos=True,
                                                         while_exposing=True))
        logger.debug("Run test getting the telescope position while exposing - COMPLETE")

        # Run for specified location
        asyncio.get_event_loop().run_until_complete(doit(get_tel_pos=False))
        logger.debug("Run for specified location - COMPLETE")

    def test_offsets(self):
        """Test offset command (which applies offsets to the models)"""

        async def doit():

            async with Harness() as harness:

                # if there is nothing (atpneumatics/atspectrograph) sending events then this command times out
                # need to extend the timeout in this case.
                await salobj.set_summary_state(harness.aos_remote, salobj.State.ENABLED, timeout=60)
                self.assertEqual(harness.csc.summary_state, salobj.State.ENABLED)

                offset_init = await harness.aos_remote.evt_correctionOffsets.next(
                    flush=False,
                    timeout=STD_TIMEOUT)

                offset = {'m1': 1.0,
                          'm2': 1.0,
                          'x': 1.0,
                          'y': 1.0,
                          'z': 1.0,
                          'u': 1.0,
                          'v': 1.0
                          }

                for axis in offset:
                    with self.subTest(axis=axis):
                        self.assertEqual(0.,
                                         getattr(offset_init, axis))

                harness.aos_remote.evt_correctionOffsets.flush()

                await harness.aos_remote.cmd_offset.set_start(**offset,
                                                              timeout=STD_TIMEOUT)

                offset_applied = await harness.aos_remote.evt_correctionOffsets.next(
                    flush=False,
                    timeout=STD_TIMEOUT)

                for axis in offset:
                    with self.subTest(axis=axis):
                        self.assertEqual(offset[axis],
                                         getattr(offset_applied, axis))

                await harness.aos_remote.cmd_resetOffset.start(timeout=STD_TIMEOUT)

                offset_reset = await harness.aos_remote.evt_correctionOffsets.next(
                    flush=False,
                    timeout=LONG_TIMEOUT)

                for axis in offset:
                    with self.subTest(axis=axis):
                        self.assertEqual(0.,
                                         getattr(offset_reset, axis))

        asyncio.get_event_loop().run_until_complete(doit())

    def test_spectrograph_offsets(self):
        """Test offsets command and handling of offsets during filter/grating changes."""

        async def doit(atspectrograph=True,
                       online_before_ataos=False,
                       correction_loop=True):

            async with Harness() as harness:

                # send pneumatics data, just speeds up the tests
                harness.pnematics.evt_summaryState.set_put(summaryState=salobj.State.ENABLED)
                harness.pnematics.evt_mainValveState.set_put(state=ATPneumatics.AirValveState.OPENED)
                harness.pnematics.evt_instrumentState.set_put(state=ATPneumatics.AirValveState.OPENED)
                harness.pnematics.evt_m1State.set_put(state=ATPneumatics.AirValveState.CLOSED)
                harness.pnematics.evt_m2State.set_put(state=ATPneumatics.AirValveState.CLOSED)

                # set the hexapod callback
                def hexapod_move_callback(data):
                    harness.hexapod.evt_positionUpdate.put()

                # Add callback to events
                def callback(data):
                    pass

                # Set required callbacks
                harness.hexapod.cmd_moveToPosition.callback = Mock(wraps=hexapod_move_callback)
                harness.aos_remote.evt_detailedState.callback = Mock(wraps=callback)

                # Can only set these if the spectrograph is not going to come online
                # if it goes offline the values will remain and no new filter/disperser
                # positions will get published
                if atspectrograph:
                    filter_name, filter_name2 = 'test_filt1', 'empty_1'
                    filter_position, filter_position2 = 1, 2
                    filter_focus_offset, filter_focus_offset2 = 0.03, 0.0
                    filter_central_wavelength, filter_central_wavelength2 = 707.0, 700

                    disperser_name, disperser_name2 = 'test_disp1', 'empty_2'
                    disperser_position, disperser_position2 = 1, 2
                    disperser_focus_offset, disperser_focus_offset2 = 0.1, 0.0

                else:
                    filter_name, filter_name2 = '', ''
                    filter_focus_offset, filter_focus_offset2 = 0.0, 0.0
                    filter_central_wavelength, filter_central_wavelength2 = 0.0, 0.0

                    disperser_name, disperser_name2 = '', ''
                    disperser_focus_offset, disperser_focus_offset2 = 0.0, 0.0

                if atspectrograph and online_before_ataos:
                    logger.debug('Loading filter and dispersers before enabling ATAOS')
                    # Bring spectrograph online and load filter/disperser
                    harness.atspectrograph.evt_summaryState.set_put(summaryState=salobj.State.ENABLED)

                    harness.atspectrograph.evt_reportedFilterPosition.set_put(
                        name=filter_name,
                        position=filter_position,
                        centralWavelength=filter_central_wavelength,
                        focusOffset=filter_focus_offset
                    )
                    harness.atspectrograph.evt_reportedDisperserPosition.set_put(
                        name=disperser_name,
                        position=disperser_position,
                        focusOffset=disperser_focus_offset
                    )
                    await asyncio.sleep(1)
                # If the atpneumatics and atspectrograph sending events then this command times out
                # when using the default timeout, therefore it is extended here to account for that
                # case.

                await salobj.set_summary_state(harness.aos_remote, salobj.State.ENABLED, timeout=60)
                self.assertEqual(harness.csc.summary_state, salobj.State.ENABLED)

                # send elevation/azimuth positions
                azimuth = np.random.uniform(0., 360.)
                # make sure it is never zero because np.random.uniform is [min, max)
                elevation = 90. - np.random.uniform(0., 90.)

                await publish_mountEncoders(harness, azimuth, elevation, ntimes=5)

                # Start the ATAOS correction loop
                # this would fail if the spectrograph isn't online!
                if correction_loop is True and online_before_ataos is True:
                    logger.debug('Enabling atspectrograph and hexapod corrections')
                    await harness.aos_remote.cmd_enableCorrection.set_start(atspectrograph=True,
                                                                            hexapod=True)
                    # let the loop turn a few times
                    await asyncio.sleep(10)

                if atspectrograph and online_before_ataos is False:
                    logger.debug('Loading filter and dispersers after enabling ATAOS')
                    # Bring spectrograph online and load filter/disperser
                    harness.atspectrograph.evt_summaryState.set_put(summaryState=salobj.State.ENABLED)

                    # the summarystate causes offsets to be published, so flush these to be sure the grab
                    # the proper one below
                    harness.aos_remote.evt_correctionOffsets.flush()
                    harness.aos_remote.evt_focusOffsetSummary.flush()

                    harness.atspectrograph.evt_reportedFilterPosition.set_put(
                        name=filter_name,
                        position=1,
                        centralWavelength=filter_central_wavelength,
                        focusOffset=filter_focus_offset
                    )
                    harness.atspectrograph.evt_reportedDisperserPosition.set_put(
                        name=disperser_name,
                        position=1,
                        focusOffset=disperser_focus_offset
                    )
                    # let the loop turn a few times otherwise it'll say the component is offline when
                    # trying to add a correction
                    await asyncio.sleep(1)

                if correction_loop is True and online_before_ataos is False:
                    logger.debug('Enabling atspectrograph and hexapod corrections')
                    await harness.aos_remote.cmd_enableCorrection.set_start(atspectrograph=True,
                                                                            hexapod=True)
                    # let the loop turn a few times
                    await asyncio.sleep(5)

                # check spectrograph corrections were applied
                correctionOffsets = await harness.aos_remote.evt_correctionOffsets.aget(timeout=STD_TIMEOUT)
                # check spectrograph accounting is being done correctly
                focusOffsetSummary = await harness.aos_remote.evt_focusOffsetSummary.aget(timeout=STD_TIMEOUT)

                if correction_loop is True:
                    logger.debug('Disabling corrections')
                    await harness.aos_remote.cmd_disableCorrection.set_start(hexapod=True,
                                                                             atspectrograph=True)

                    self.assertAlmostEqual(correctionOffsets.z, filter_focus_offset + disperser_focus_offset)
                    self.assertAlmostEqual(focusOffsetSummary.total,
                                           disperser_focus_offset + filter_focus_offset)
                else:
                    # offsets from filter/dispersers should not yet be applied if correction loop isn't on
                    self.assertAlmostEqual(correctionOffsets.z, 0.0)
                    self.assertAlmostEqual(focusOffsetSummary.total,
                                           0.0)

                self.assertAlmostEqual(focusOffsetSummary.userApplied, 0.0)
                self.assertAlmostEqual(focusOffsetSummary.filter, filter_focus_offset)
                self.assertAlmostEqual(focusOffsetSummary.disperser, disperser_focus_offset)

                offset = {'m1': 1.1,
                          'm2': 1.2,
                          'x': 1.3,
                          'y': 1.4,
                          'z': 1.5,
                          'u': 1.6,
                          'v': 1.7
                          }

                # Now start the loops, we'll then add an offset, then remove a filter, then remove
                # a disperser
                if correction_loop is True:
                    logger.debug('Re-enabling atspectrograph and hexapod corrections')
                    await harness.aos_remote.cmd_enableCorrection.set_start(atspectrograph=True,
                                                                            hexapod=True)

                # flush events then send relative offsets
                harness.aos_remote.evt_correctionOffsets.flush()
                harness.aos_remote.evt_focusOffsetSummary.flush()

                # add the userApplied-offset
                await harness.aos_remote.cmd_offset.set_start(**offset,
                                                              timeout=STD_TIMEOUT)

                offset_applied = await harness.aos_remote.evt_correctionOffsets.next(flush=False,
                                                                                     timeout=STD_TIMEOUT)
                focusOffsetSummary = await harness.aos_remote.evt_focusOffsetSummary.next(flush=False,
                                                                                          timeout=STD_TIMEOUT)

                # offsets should be combined in z
                for axis in offset:
                    logger.debug('axis = {} and correction_offset'
                                 ' = {}'.format(axis, getattr(offset_applied, axis)))
                    with self.subTest(axis=axis):
                        if axis != 'z':
                            self.assertAlmostEqual(offset[axis],
                                                   getattr(offset_applied, axis))
                        else:
                            # should be applied offset plus the filter/disperser offsets
                            self.assertAlmostEqual(
                                offset[axis] + disperser_focus_offset + filter_focus_offset,
                                getattr(offset_applied, axis))

                # check that summary is correct
                self.assertAlmostEqual(focusOffsetSummary.total,
                                       getattr(offset_applied, 'z'))
                # userApplied offset should just be whatever we supplied
                self.assertAlmostEqual(focusOffsetSummary.userApplied, offset['z'])
                self.assertAlmostEqual(focusOffsetSummary.filter, filter_focus_offset)
                self.assertAlmostEqual(focusOffsetSummary.disperser, disperser_focus_offset)

                # This part of the test is only applicable if the spectrograph is online
                if atspectrograph:
                    logger.debug('Putting in filter2')
                    # flush events then change filters
                    harness.aos_remote.evt_correctionOffsets.flush()
                    harness.aos_remote.evt_focusOffsetSummary.flush()

                    harness.atspectrograph.evt_reportedFilterPosition.set_put(
                        name=filter_name2,
                        position=filter_position2,
                        centralWavelength=filter_central_wavelength2,
                        focusOffset=filter_focus_offset2
                    )

                    offset_applied = await harness.aos_remote.evt_correctionOffsets.next(
                        flush=False,
                        timeout=STD_TIMEOUT
                    )
                    focusOffsetSummary = await harness.aos_remote.evt_focusOffsetSummary.next(
                        flush=False,
                        timeout=STD_TIMEOUT
                    )

                    # offsets should be combined in z
                    # this could be made a function, but I found this easier to read/parse/understand
                    for axis in offset:

                        with self.subTest(axis=axis):
                            if axis != 'z':
                                self.assertAlmostEqual(offset[axis],
                                                       getattr(offset_applied, axis))
                            else:
                                # should be applied offset plus the filter/disperser offsets
                                self.assertAlmostEqual(
                                    offset[axis] + disperser_focus_offset + filter_focus_offset2,
                                    getattr(offset_applied, axis))

                    # check that summary is correct
                    self.assertAlmostEqual(focusOffsetSummary.total,
                                           getattr(offset_applied, 'z'))
                    # userApplied offset should just be whatever we supplied
                    self.assertAlmostEqual(focusOffsetSummary.userApplied, offset['z'])
                    self.assertAlmostEqual(focusOffsetSummary.filter, filter_focus_offset2)
                    self.assertAlmostEqual(focusOffsetSummary.disperser, disperser_focus_offset)

                    # flush events then change dispersers
                    logger.debug('Putting in disperser2')
                    harness.aos_remote.evt_correctionOffsets.flush()
                    harness.aos_remote.evt_focusOffsetSummary.flush()

                    harness.atspectrograph.evt_reportedDisperserPosition.set_put(
                        name=disperser_name2,
                        position=disperser_position2,
                        focusOffset=disperser_focus_offset2
                    )

                    offset_applied = await harness.aos_remote.evt_correctionOffsets.next(
                        flush=False,
                        timeout=STD_TIMEOUT
                    )
                    focusOffsetSummary = await harness.aos_remote.evt_focusOffsetSummary.next(
                        flush=False,
                        timeout=STD_TIMEOUT
                    )

                    # offsets should be combined in z
                    for axis in offset:

                        with self.subTest(axis=axis):
                            if axis != 'z':
                                self.assertAlmostEqual(offset[axis],
                                                       getattr(offset_applied, axis))
                            else:
                                # should be applied offset plus the filter/disperser offsets
                                self.assertAlmostEqual(
                                    offset[axis] + disperser_focus_offset2 + filter_focus_offset2,
                                    getattr(offset_applied, axis))

                    # check that summary is correct
                    self.assertAlmostEqual(focusOffsetSummary.total,
                                           getattr(offset_applied, 'z'))
                    # userApplied offset should just be whatever we supplied
                    self.assertAlmostEqual(focusOffsetSummary.userApplied, offset['z'])
                    self.assertAlmostEqual(focusOffsetSummary.filter, filter_focus_offset2)
                    self.assertAlmostEqual(focusOffsetSummary.disperser, disperser_focus_offset2)

                # Now reset the offsets (after flushing events)
                # will not reset spectrograph offsets!
                harness.aos_remote.evt_correctionOffsets.flush()
                harness.aos_remote.evt_focusOffsetSummary.flush()
                await harness.aos_remote.cmd_resetOffset.start(timeout=STD_TIMEOUT)

                offset_applied = await harness.aos_remote.evt_correctionOffsets.next(
                    flush=False,
                    timeout=STD_TIMEOUT)
                focusOffsetSummary = \
                    await harness.aos_remote.evt_focusOffsetSummary.next(flush=False, timeout=STD_TIMEOUT)

                for axis in offset:

                    with self.subTest(axis=axis):
                        if axis != 'z':
                            self.assertAlmostEqual(0.0, getattr(offset_applied, axis))
                        else:
                            # correction offset should be zero plus the filter/disperser offsets
                            self.assertAlmostEqual(0.0 + disperser_focus_offset2 + filter_focus_offset2,
                                                   getattr(offset_applied, axis))
                # totals should be just filter/disperser offsets
                self.assertAlmostEqual(focusOffsetSummary.total,
                                       disperser_focus_offset2 + filter_focus_offset2)
                # userApplied offset should be zero
                self.assertAlmostEqual(focusOffsetSummary.userApplied, 0.0)
                self.assertAlmostEqual(focusOffsetSummary.filter, filter_focus_offset2)
                self.assertAlmostEqual(focusOffsetSummary.disperser, disperser_focus_offset2)

                # Disable corrections gracefully, makes debugging easier
                if correction_loop is True:
                    logger.debug('Disabling corrections')
                    await harness.aos_remote.cmd_disableCorrection.set_start(hexapod=True,
                                                                             atspectrograph=True)

                # send spectrograph offline
                harness.atspectrograph.evt_summaryState.set_put(summaryState=salobj.State.OFFLINE)

        logger.debug('Running test with spectrograph online before ATAOS')
        asyncio.get_event_loop().run_until_complete(doit(atspectrograph=True,
                                                         online_before_ataos=True,
                                                         correction_loop=True))
        logger.debug('COMPLETED test with spectrograph online before ATAOS \n')

        logger.debug('Running test with spectrograph online after ATAOS')
        asyncio.get_event_loop().run_until_complete(doit(atspectrograph=True,
                                                         online_before_ataos=False,
                                                         correction_loop=True))
        logger.debug('COMPLETED test with spectrograph online after ATAOS \n')

        logger.debug('Running test with spectrograph offline')
        asyncio.get_event_loop().run_until_complete(doit(atspectrograph=False,
                                                         online_before_ataos=False,
                                                         correction_loop=False))
        logger.debug('COMPLETED test with spectrograph offline \n')

    def test_enable_disable_corrections(self):
        """Test enableCorrection"""

        async def doit():

            async with Harness() as harness:

                def callback(data):
                    pass

                def m1_open_callback(data):
                    harness.pnematics.evt_m1State.set_put(state=ATPneumatics.AirValveState.OPENED)

                def m2_open_callback(data):
                    harness.pnematics.evt_m2State.set_put(state=ATPneumatics.AirValveState.OPENED)

                def m1_close_callback(data):
                    harness.pnematics.evt_m1State.set_put(state=ATPneumatics.AirValveState.CLOSED)

                def m2_close_callback(data):
                    harness.pnematics.evt_m2State.set_put(state=ATPneumatics.AirValveState.CLOSED)

                harness.pnematics.cmd_m1SetPressure.callback = Mock(wraps=callback)
                harness.pnematics.cmd_m2SetPressure.callback = Mock(wraps=callback)
                harness.pnematics.cmd_m1OpenAirValve.callback = Mock(wraps=m1_open_callback)
                harness.pnematics.cmd_m2OpenAirValve.callback = Mock(wraps=m2_open_callback)
                harness.pnematics.cmd_m1CloseAirValve.callback = Mock(wraps=m1_close_callback)
                harness.pnematics.cmd_m2CloseAirValve.callback = Mock(wraps=m2_close_callback)

                harness.pnematics.evt_summaryState.set_put(summaryState=salobj.State.ENABLED)
                harness.pnematics.evt_mainValveState.set_put(state=ATPneumatics.AirValveState.OPENED)
                harness.pnematics.evt_instrumentState.set_put(state=ATPneumatics.AirValveState.OPENED)
                harness.pnematics.evt_m1State.set_put(state=ATPneumatics.AirValveState.CLOSED)
                harness.pnematics.evt_m2State.set_put(state=ATPneumatics.AirValveState.CLOSED)

                # report atspectrograph status
                harness.atspectrograph.evt_summaryState.set_put(summaryState=salobj.State.ENABLED)
                harness.atspectrograph.evt_reportedFilterPosition.set_put(name='empty_1',
                                                                          position=1,
                                                                          centralWavelength=701.5,
                                                                          focusOffset=0.0)
                harness.atspectrograph.evt_reportedDisperserPosition.set_put(name='empty_1',
                                                                             position=1,
                                                                             focusOffset=0.0)

                # extend the timeout, to account for if all events aren't set yet
                await salobj.set_summary_state(harness.aos_remote, salobj.State.ENABLED, timeout=60)
                self.assertEqual(harness.csc.summary_state, salobj.State.ENABLED)

                cmd_attr = getattr(harness.aos_remote, f"cmd_enableCorrection")

                timeout = 5 * salobj.base_csc.HEARTBEAT_INTERVAL

                # Try to send empty enable correction, this will fail.
                logger.debug('Try to send empty enable correction, this will fail.')
                with self.assertRaises(salobj.AckError):
                    await cmd_attr.start(cmd_attr.DataType(),
                                         timeout=timeout)

                # Enable corrections one by one
                corrections = ("m1", "m2", "hexapod", "focus", "moveWhileExposing", "atspectrograph")
                expected_corrections = {"m1": False, "m2": False, "hexapod": False, "focus": False,
                                        "moveWhileExposing": False, "atspectrograph": False}

                logger.debug('Starting to loop over enabling corrections')
                for corr in corrections:
                    send_topic = cmd_attr.DataType()
                    expected_corrections[corr] = True
                    # Note: Setting only the correction I want to enable. Any correction already
                    # enabled will be left unchanged. This is part of the test.
                    setattr(send_topic, corr, True)
                    logger.debug(corr)
                    coro = getattr(harness.aos_remote,
                                   f"evt_correctionEnabled").next(flush=False,
                                                                  timeout=timeout)
                    await cmd_attr.start(send_topic,
                                         timeout=timeout)
                    logger.debug(f"TEST 1: {corr}")
                    receive_topic = await coro
                    logger.debug(f"TEST 2: {corr}")
                    if corr == "m1":
                        harness.pnematics.cmd_m1SetPressure.callback.assert_called()
                        harness.pnematics.cmd_m1OpenAirValve.callback.assert_called()
                    elif corr == "m2":
                        harness.pnematics.cmd_m2SetPressure.callback.assert_called()
                        harness.pnematics.cmd_m2OpenAirValve.callback.assert_called()

                    for test_corr in corrections:
                        with self.subTest(test_corr=test_corr):
                            self.assertEqual(getattr(receive_topic, test_corr),
                                             expected_corrections[test_corr],
                                             f"Failed to set correction for {test_corr}. "
                                             f"Expected {expected_corrections[test_corr]}, "
                                             f"got {getattr(receive_topic, test_corr)}")

                cmd_attr = getattr(harness.aos_remote, f"cmd_disableCorrection")

                # Try to send empty disable correction, this will fail.
                logger.debug('Try to send empty disable correction, this will fail')
                with self.assertRaises(salobj.AckError):
                    await cmd_attr.start(cmd_attr.DataType(),
                                         timeout=timeout)

                # Disable corrections one by one
                for corr in corrections:
                    send_topic = cmd_attr.DataType()
                    expected_corrections[corr] = False  # mark as expected to False
                    # Note: Setting only the correction I want to disable. Any correction
                    # already disable will be left unchanged. This is part of the test.
                    # True means I want to disable
                    setattr(send_topic, corr, True)

                    coro = getattr(harness.aos_remote,
                                   f"evt_correctionEnabled").next(flush=False,
                                                                  timeout=timeout)
                    await cmd_attr.start(send_topic,
                                         timeout=timeout)
                    receive_topic = await coro
                    for test_corr in corrections:
                        with self.subTest(test_corr=test_corr):
                            self.assertEqual(getattr(receive_topic, test_corr),
                                             expected_corrections[test_corr],
                                             f"Failed to set correction for {test_corr}. "
                                             f"Expected {expected_corrections[test_corr]}, "
                                             f"got {getattr(receive_topic, test_corr)}")

                # everything is disable, send enable all.
                logger.debug('everything is disable, send enable all.')
                cmd_attr = getattr(harness.aos_remote, f"cmd_enableCorrection")
                send_topic = cmd_attr.DataType()
                send_topic.enableAll = True

                coro = getattr(harness.aos_remote,
                               f"evt_correctionEnabled").next(flush=False,
                                                              timeout=timeout)
                await cmd_attr.start(send_topic,
                                     timeout=timeout)
                receive_topic = await coro

                # They should all be true, except moveWhileExposing
                for test_corr in corrections:
                    with self.subTest(test_corr=test_corr):
                        if test_corr == 'moveWhileExposing':
                            self.assertFalse(getattr(receive_topic, test_corr),
                                             f"Failed to set correction for moveWhileExposing. "
                                             f"Expected True, "
                                             f"got {getattr(receive_topic, test_corr)}")
                        else:
                            self.assertTrue(getattr(receive_topic, test_corr),
                                            f"Failed to set correction for {test_corr}. "
                                            f"Expected True, "
                                            f"got {getattr(receive_topic, test_corr)}")

                # everything is enable, send disable all
                logger.debug('everything is enable, send disable all.')
                cmd_attr = getattr(harness.aos_remote, f"cmd_disableCorrection")
                send_topic = cmd_attr.DataType()
                send_topic.disableAll = True

                coro = getattr(harness.aos_remote,
                               f"evt_correctionEnabled").next(flush=False,
                                                              timeout=timeout)
                await cmd_attr.start(send_topic,
                                     timeout=timeout)
                receive_topic = await coro

                # They should all be False
                for test_corr in corrections:
                    with self.subTest(test_corr=test_corr):
                        self.assertFalse(getattr(receive_topic, test_corr),
                                         f"Failed to set correction for {test_corr}. "
                                         f"Expected False, "
                                         f"got {getattr(receive_topic, test_corr)}")

        logger.debug('Starting test_enable_disable_corrections')
        asyncio.get_event_loop().run_until_complete(doit())


async def publish_mountEncoders(harness, azimuth, elevation, ntimes=5):
    """Publish telescope position as an event"""

    # arrays need to have a length of 100 values
    for i in range(ntimes):
        _azimuth = np.zeros(100) + azimuth
        # make sure it is never zero because np.random.uniform is [min, max)
        _elevation = np.zeros(100) + elevation
        harness.atmcs.tel_mount_AzEl_Encoders.set_put(
            azimuthCalculatedAngle=_azimuth,
            elevationCalculatedAngle=_elevation)
        await asyncio.sleep(salobj.base_csc.HEARTBEAT_INTERVAL)


if __name__ == '__main__':
    stream_handler = logging.StreamHandler(sys.stdout)  # leave uncommented to prevent flake8 error
    # stream_handler.setLevel(logging.DEBUG)
    # logger.addHandler(stream_handler)

    unittest.main()
