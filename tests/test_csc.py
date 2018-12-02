import unittest
from unittest.mock import Mock
import asyncio
import numpy as np

from lsst.ts import salobj

from lsst.ts.ataos import ataos_csc

import SALPY_ATAOS

import SALPY_ATMCS
import SALPY_ATPneumatics
import SALPY_ATHexapod

np.random.seed(47)

index_gen = salobj.index_generator()


class Harness:
    def __init__(self):
        salobj.test_utils.set_random_lsst_dds_domain()
        self.csc = ataos_csc.ATAOS()

        # Adds a remote to control the ATAOS CSC
        self.aos_remote = salobj.Remote(SALPY_ATAOS)

        # Adds Controllers to receive commands from the ATAOS system
        self.atmcs = salobj.Controller(SALPY_ATMCS)
        self.pnematics = salobj.Controller(SALPY_ATPneumatics)
        self.hexapod = salobj.Controller(SALPY_ATHexapod)

    async def enable_csc(self):
        """Utility method to enable the Harness csc."""

        commands = ("start", "enable")

        for cmd in commands:
            cmd_attr = getattr(self.aos_remote, f"cmd_{cmd}")
            await cmd_attr.start(cmd_attr.DataType(), timeout=5*salobj.base_csc.HEARTBEAT_INTERVAL)


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

            harness = Harness()

            # Check initial state
            current_state = await harness.aos_remote.evt_summaryState.next(flush=False, timeout=1.)

            self.assertEqual(harness.csc.summary_state, salobj.State.STANDBY)
            self.assertEqual(current_state.summaryState, salobj.State.STANDBY)

            # Check that settingVersions was published and matches expected values
            setting_versions = await harness.aos_remote.evt_settingVersions.next(flush=False, timeout=1.)
            self.assertEqual(setting_versions.recommendedSettingsVersion,
                             harness.csc.model.recommended_settings)
            self.assertEqual(setting_versions.recommendedSettingsLabels,
                             harness.csc.model.settings_labels)
            self.assertTrue(setting_versions.recommendedSettingsVersion in
                            setting_versions.recommendedSettingsLabels.split(','))
            self.assertTrue('test' in
                            setting_versions.recommendedSettingsLabels.split(','))

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
            state_coro = harness.aos_remote.evt_summaryState.next(flush=True, timeout=1.)
            start_topic = cmd_attr.DataType()
            start_topic.settingsToApply = 'test'  # test settings.
            id_ack = await cmd_attr.start(start_topic, timeout=120)  # this one can take longer to execute
            state = await state_coro
            self.assertEqual(id_ack.ack.ack, harness.aos_remote.salinfo.lib.SAL__CMD_COMPLETE)
            self.assertEqual(id_ack.ack.error, 0)
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
            state_coro = harness.aos_remote.evt_summaryState.next(flush=True, timeout=1.)
            id_ack = await cmd_attr.start(cmd_attr.DataType(), timeout=1.)
            state = await state_coro
            self.assertEqual(id_ack.ack.ack, harness.aos_remote.salinfo.lib.SAL__CMD_COMPLETE)
            self.assertEqual(id_ack.ack.error, 0)
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
            self.assertEqual(id_ack.ack.ack, harness.aos_remote.salinfo.lib.SAL__CMD_COMPLETE)
            self.assertEqual(id_ack.ack.error, 0)
            self.assertEqual(harness.csc.summary_state, salobj.State.DISABLED)

        asyncio.get_event_loop().run_until_complete(doit())

    def test_applyCorrection(self):
        """Test applyCorrection command. """

        async def doit():
            harness = Harness()
            timeout = 5 * salobj.base_csc.HEARTBEAT_INTERVAL
            # Enable the CSC
            await harness.enable_csc()
            self.assertEqual(harness.csc.summary_state, salobj.State.ENABLED)

            # Check that applyCorrection fails if enable Correction is on
            cmd_attr = getattr(harness.aos_remote, f"cmd_enableCorrection")

            send_topic = cmd_attr.DataType()
            send_topic.m1 = True

            await cmd_attr.start(send_topic,
                                 timeout=timeout)

            cmd_attr = getattr(harness.aos_remote, f"cmd_applyCorrection")
            with self.assertRaises(salobj.AckError):
                await cmd_attr.start(cmd_attr.DataType(),
                                     timeout=timeout)

            # Switch corrections off
            cmd_attr = getattr(harness.aos_remote, f"cmd_disableCorrection")
            send_topic = cmd_attr.DataType()
            send_topic.all = True

            await cmd_attr.start(send_topic,
                                 timeout=timeout)

            # Check that applyCorrection works for current telescope position
            def callback(data):
                pass

            # Add callback to commands from pneumatics and hexapod
            harness.pnematics.cmd_m1SetPressure.callback = Mock(wraps=callback)
            harness.pnematics.cmd_m2SetPressure.callback = Mock(wraps=callback)

            # FIXME: Check if this is correct! Is there a difference in command to move hexapod and focus
            # or they will use the same command?
            harness.hexapod.cmd_moveToPosition.callback = Mock(wraps=callback)

            # Add callback to events
            harness.aos_remote.evt_detailedState.callback = Mock(wraps=callback)

            harness.aos_remote.evt_m1CorrectionStarted.callback = Mock(wraps=callback)
            harness.aos_remote.evt_m2CorrectionStarted.callback = Mock(wraps=callback)
            harness.aos_remote.evt_hexapodCorrectionStarted.callback = Mock(wraps=callback)

            harness.aos_remote.evt_m1CorrectionCompleted.callback = Mock(wraps=callback)
            harness.aos_remote.evt_m2CorrectionCompleted.callback = Mock(wraps=callback)
            harness.aos_remote.evt_hexapodCorrectionCompleted.callback = Mock(wraps=callback)

            # Publish telescope position using atmcs controller from harness
            azimuth = 0.  # Test azimuth = 0.
            # make sure it is never zero because np.random.uniform is [min, max)
            elevation = 90.-np.random.uniform(0., 90.)

            async def publish_mountEnconders(az, el, ntimes=5):
                topic = harness.atmcs.tel_mountEncoders.DataType()
                topic.azimuthCalculatedAngle = az
                topic.elevationCalculatedAngle = el
                for i in range(ntimes):
                    await asyncio.sleep(salobj.base_csc.HEARTBEAT_INTERVAL)
                    harness.atmcs.tel_mountEncoders.put(topic)

            # Send applyCorrection command using default values, should get the position from the
            # tel_mountEncoders telemetry.
            cmd_attr = getattr(harness.aos_remote, f"cmd_applyCorrection")

            await asyncio.gather(cmd_attr.start(cmd_attr.DataType(),
                                                timeout=timeout),
                                 publish_mountEnconders(azimuth, elevation))

            # Give control back to event loop so it can gather remaining callbacks
            await asyncio.sleep(5*salobj.base_csc.HEARTBEAT_INTERVAL)

            # Check that callbacks where called
            harness.pnematics.cmd_m1SetPressure.callback.assert_called()
            harness.pnematics.cmd_m2SetPressure.callback.assert_called()
            harness.hexapod.cmd_moveToPosition.callback.assert_called()
            harness.aos_remote.evt_detailedState.callback.assert_called()
            harness.aos_remote.evt_m1CorrectionStarted.callback.assert_called()
            harness.aos_remote.evt_m2CorrectionStarted.callback.assert_called()
            harness.aos_remote.evt_hexapodCorrectionStarted.callback.assert_called()
            harness.aos_remote.evt_m1CorrectionCompleted.callback.assert_called()
            harness.aos_remote.evt_m2CorrectionCompleted.callback.assert_called()
            harness.aos_remote.evt_hexapodCorrectionCompleted.callback.assert_called()

            self.assertEqual(len(harness.aos_remote.evt_detailedState.callback.call_args_list),
                             6,
                             '%s' % harness.aos_remote.evt_detailedState.callback.call_args_list)

        asyncio.get_event_loop().run_until_complete(doit())

    def test_applyFocusOffset(self):
        """Test applyFocusOffset command."""
        pass

    def test_enable_disable_corrections(self):
        """Test enableCorrection"""

        async def doit():
            harness = Harness()

            # Enable the CSC
            await harness.enable_csc()
            self.assertEqual(harness.csc.summary_state, salobj.State.ENABLED)

            cmd_attr = getattr(harness.aos_remote, f"cmd_enableCorrection")
            # id_ack = await cmd_attr.start(cmd_attr.DataType(), timeout=5*salobj.base_csc.HEARTBEAT_INTERVAL)

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
                                         "Failed to set correction for %s. "
                                         "Expected %s, got %s" % (test_corr,
                                                                  expected_corrections[test_corr],
                                                                  getattr(receive_topic, test_corr)))

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
                                         "Failed to set correction for %s. "
                                         "Expected %s, got %s" % (test_corr,
                                                                  expected_corrections[test_corr],
                                                                  getattr(receive_topic, test_corr)))

            # everything is disable, send enable all.
            cmd_attr = getattr(harness.aos_remote, f"cmd_enableCorrection")
            send_topic = cmd_attr.DataType()
            send_topic.all = True

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
                                         "Failed to set correction for moveWhileExposing. "
                                         "Expected True, got %s" % getattr(receive_topic, test_corr))
                    else:
                        self.assertTrue(getattr(receive_topic, test_corr),
                                        "Failed to set correction for %s. "
                                        "Expected %s, got %s" % (test_corr,
                                                                 "True",
                                                                 getattr(receive_topic, test_corr)))

            # everything is enable, send disable all
            cmd_attr = getattr(harness.aos_remote, f"cmd_disableCorrection")
            send_topic = cmd_attr.DataType()
            send_topic.all = True

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
                                     "Failed to set correction for %s. "
                                     "Expected %s, got %s" % (test_corr,
                                                              "False",
                                                              getattr(receive_topic, test_corr)))

        asyncio.get_event_loop().run_until_complete(doit())

    def test_setFocus(self):
        """Test setFocus"""
        pass


if __name__ == '__main__':
    unittest.main()
