import sys
import unittest
from unittest.mock import Mock
import asyncio
import numpy as np
import logging

from lsst.ts import salobj

from lsst.ts.ataos import ataos_csc

from lsst.ts.idl.enums import ATPneumatics

np.random.seed(47)

index_gen = salobj.index_generator()

logger = logging.getLogger()
logger.level = logging.DEBUG


class Harness:
    def __init__(self):
        salobj.test_utils.set_random_lsst_dds_domain()
        self.csc = ataos_csc.ATAOS()

        # Adds a remote to control the ATAOS CSC
        self.aos_remote = salobj.Remote(self.csc.domain, "ATAOS")

        # Adds Controllers to receive commands from the ATAOS system
        self.atmcs = salobj.Controller("ATMCS")
        self.pnematics = salobj.Controller("ATPneumatics")
        self.hexapod = salobj.Controller("ATHexapod")
        self.camera = salobj.Controller("ATCamera")

    async def __aenter__(self):
        await asyncio.gather(self.csc.start_task,
                             self.aos_remote.start_task,
                             self.atmcs.start_task,
                             self.pnematics.start_task,
                             self.hexapod.start_task,
                             self.camera.start_task)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await asyncio.gather(self.csc.close(),
                             self.aos_remote.close(),
                             self.atmcs.close(),
                             self.pnematics.close(),
                             self.hexapod.close(),
                             self.camera.close())


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

            extra_commands = ("applyCorrection",
                              "applyFocusOffset",
                              "disableCorrection",
                              "enableCorrection",
                              "setFocus")

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
        """Test applyCorrection command. """

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

                # FIXME: Check if this is correct! Is there a difference in
                # command to move hexapod and focus or they will use the same
                # command?
                harness.hexapod.cmd_moveToPosition.callback = Mock(wraps=hexapod_move_callback)

                # Add callback to events
                harness.aos_remote.evt_detailedState.callback = Mock(wraps=callback)

                timeout = 40. * salobj.base_csc.HEARTBEAT_INTERVAL
                # Enable the CSC
                await salobj.set_summary_state(harness.aos_remote, salobj.State.ENABLED)
                self.assertEqual(harness.csc.summary_state, salobj.State.ENABLED)

                # Check that applyCorrection fails if enable Correction is on
                harness.aos_remote.cmd_enableCorrection.set(m1=True)
                await harness.aos_remote.cmd_enableCorrection.start(timeout=timeout)

                with self.assertRaises(salobj.AckError):
                    await harness.aos_remote.cmd_applyCorrection.start(timeout=timeout)

                # Switch corrections off
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

                # Send applyCorrection command
                azimuth = np.random.uniform(0., 360.)
                # make sure it is never zero because np.random.uniform is [min, max)
                elevation = 90. - np.random.uniform(0., 90.)

                async def publish_mountEnconders(ntimes=5):

                    for i in range(ntimes):
                        _azimuth = np.zeros(100)+azimuth
                        # make sure it is never zero because np.random.uniform is [min, max)
                        _elevation = np.zeros(100)+elevation
                        harness.atmcs.tel_mount_AzEl_Encoders.set_put(
                            azimuthCalculatedAngle=_azimuth,
                            elevationCalculatedAngle=_elevation)
                        await asyncio.sleep(salobj.base_csc.HEARTBEAT_INTERVAL)

                await publish_mountEnconders()

                # Test that the hexapod won't move if there's an exposure happening
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
                await asyncio.sleep(5*salobj.base_csc.HEARTBEAT_INTERVAL)

                # Check that callbacks where called
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

                # disable CSC
                await harness.aos_remote.cmd_disable.start(timeout=timeout)

        # Run test getting the telescope position
        asyncio.get_event_loop().run_until_complete(doit(get_tel_pos=True))

        # Run test getting the telescope position while exposing
        asyncio.get_event_loop().run_until_complete(doit(get_tel_pos=True,
                                                         while_exposing=True))

        # Run for specified location
        asyncio.get_event_loop().run_until_complete(doit(get_tel_pos=False))

    def test_applyFocusOffset(self):
        """Test applyFocusOffset command."""
        pass

    def test_enable_disable_corrections(self):
        """Test enableCorrection"""

        async def doit():

            async with Harness() as harness:

                # Enable the CSC
                await salobj.set_summary_state(harness.aos_remote, salobj.State.ENABLED)
                self.assertEqual(harness.csc.summary_state, salobj.State.ENABLED)

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

                cmd_attr = getattr(harness.aos_remote, f"cmd_enableCorrection")

                timeout = 5 * salobj.base_csc.HEARTBEAT_INTERVAL

                # Try to send empty enable correction, this will fail.
                with self.assertRaises(salobj.AckError):
                    await cmd_attr.start(cmd_attr.DataType(),
                                         timeout=timeout)

                # Enable corrections one by one
                corrections = ("m1", "m2", "hexapod", "focus", "moveWhileExposing")
                expected_corrections = {"m1": False, "m2": False, "hexapod": False, "focus": False,
                                        "moveWhileExposing": False}

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

        asyncio.get_event_loop().run_until_complete(doit())

    def test_setFocus(self):
        """Test setFocus"""
        pass


if __name__ == '__main__':

    stream_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(stream_handler)

    unittest.main()
