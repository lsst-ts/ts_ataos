import unittest
import asyncio
import numpy as np

from lsst.ts import salobj

from lsst.ts.ataos import ataos_csc

import SALPY_ATAOS

np.random.seed(47)

index_gen = salobj.index_generator()


class Harness:
    def __init__(self):
        salobj.test_utils.set_random_lsst_dds_domain()
        self.csc = ataos_csc.ATAOS()
        self.remote = salobj.Remote(SALPY_ATAOS)


class TestDIMMCSC(unittest.TestCase):

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
            current_state = await harness.remote.evt_summaryState.next(flush=False, timeout=1.)

            self.assertEqual(harness.csc.summary_state, salobj.State.STANDBY)
            self.assertEqual(current_state.summaryState, salobj.State.STANDBY)

            # Check that settingVersions was published and matches expected values
            setting_versions = await harness.remote.evt_settingVersions.next(flush=False, timeout=1.)
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
                    cmd_attr = getattr(harness.remote, f"cmd_{bad_command}")
                    with self.assertRaises(salobj.AckError):
                        id_ack = await cmd_attr.start(cmd_attr.DataType(), timeout=1.)

            for bad_command in extra_commands:
                with self.subTest(bad_command=bad_command):
                    cmd_attr = getattr(harness.remote, f"cmd_{bad_command}")
                    with self.assertRaises(salobj.AckError):
                        id_ack = await cmd_attr.start(cmd_attr.DataType(), timeout=1.)

            # send start; new state is DISABLED
            cmd_attr = getattr(harness.remote, f"cmd_start")
            state_coro = harness.remote.evt_summaryState.next(timeout=1.)
            start_topic = cmd_attr.DataType()
            start_topic.settingsToApply = 'test'  # test settings.
            id_ack = await cmd_attr.start(start_topic, timeout=120)  # this one can take longer to execute
            state = await state_coro
            self.assertEqual(id_ack.ack.ack, harness.remote.salinfo.lib.SAL__CMD_COMPLETE)
            self.assertEqual(id_ack.ack.error, 0)
            self.assertEqual(harness.csc.summary_state, salobj.State.DISABLED)
            self.assertEqual(state.summaryState, salobj.State.DISABLED)

            # TODO: There are two events issued when starting; appliedSettingsMatchStart and
            # settingsApplied. Check that they are received.

            for bad_command in commands:
                if bad_command in ("enable", "standby"):
                    continue  # valid command in DISABLED state
                with self.subTest(bad_command=bad_command):
                    cmd_attr = getattr(harness.remote, f"cmd_{bad_command}")
                    with self.assertRaises(salobj.AckError):
                        id_ack = await cmd_attr.start(cmd_attr.DataType(), timeout=1.)

            for bad_command in extra_commands:
                with self.subTest(bad_command=bad_command):
                    cmd_attr = getattr(harness.remote, f"cmd_{bad_command}")
                    with self.assertRaises(salobj.AckError):
                        id_ack = await cmd_attr.start(cmd_attr.DataType(), timeout=1.)

            # send enable; new state is ENABLED
            cmd_attr = getattr(harness.remote, f"cmd_enable")
            state_coro = harness.remote.evt_summaryState.next(timeout=1.)
            id_ack = await cmd_attr.start(cmd_attr.DataType(), timeout=1.)
            state = await state_coro
            self.assertEqual(id_ack.ack.ack, harness.remote.salinfo.lib.SAL__CMD_COMPLETE)
            self.assertEqual(id_ack.ack.error, 0)
            self.assertEqual(harness.csc.summary_state, salobj.State.ENABLED)
            self.assertEqual(state.summaryState, salobj.State.ENABLED)

            for bad_command in commands:
                if bad_command == "disable":
                    continue  # valid command in ENABLE state
                with self.subTest(bad_command=bad_command):
                    cmd_attr = getattr(harness.remote, f"cmd_{bad_command}")
                    with self.assertRaises(salobj.AckError):
                        id_ack = await cmd_attr.start(cmd_attr.DataType(), timeout=1.)

            # Now, check that the commands works
            for good_command in extra_commands:
                with self.subTest(bad_command=good_command):
                    cmd_attr = getattr(harness.remote, f"cmd_{good_command}")
                    try:
                        id_ack = await cmd_attr.start(cmd_attr.DataType(), timeout=1.)
                    except salobj.AckError:
                        self.assertTrue(False, f"Command {good_command} failed")

            # send disable; new state is DISABLED
            cmd_attr = getattr(harness.remote, f"cmd_disable")
            # this CMD may take some time to complete
            id_ack = await cmd_attr.start(cmd_attr.DataType(), timeout=30.)
            self.assertEqual(id_ack.ack.ack, harness.remote.salinfo.lib.SAL__CMD_COMPLETE)
            self.assertEqual(id_ack.ack.error, 0)
            self.assertEqual(harness.csc.summary_state, salobj.State.DISABLED)

        asyncio.get_event_loop().run_until_complete(doit())


if __name__ == '__main__':
    unittest.main()
