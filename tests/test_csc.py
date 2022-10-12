# This file is part of ts_observatory_control.
#
# Developed for the Vera Rubin Observatory Telescope and Site Systems.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License

import asyncio
import contextlib
import logging
import pathlib
import typing
import unittest
import unittest.mock

import numpy as np
import pytest
from lsst.ts.idl.enums import ATPneumatics

from lsst.ts import ataos, salobj

STD_TIMEOUT = 5  # standard command timeout (sec)
LONG_TIMEOUT = 20  # timeout for starting SAL components (sec)
TEST_CONFIG_DIR = pathlib.Path(__file__).parents[1].joinpath("tests", "data", "config")


class TestCSC(salobj.BaseCscTestCase, unittest.IsolatedAsyncioTestCase):
    def basic_make_csc(
        self,
        initial_state: typing.Union[salobj.sal_enums.State, int],
        config_dir: typing.Union[str, pathlib.Path, None],
        simulation_mode: int,
    ) -> salobj.base_csc.BaseCsc:
        return ataos.ATAOS(config_dir=config_dir)

    def setUp(self) -> None:

        self.log = logging.getLogger(type(self).__name__)

        self._telescope_azimuth = 0.0
        self._telescope_elevation = 80.0

        # provide spectrograph setup info
        self.filter_name = "test_filt1"
        self.filter_focus_offset = 0.03
        self.filter_central_wavelength = 707.0
        self.filter_pointing_offsets = np.array([0.1, -0.1])

        self.disperser_name = "test_disp1"
        self.disperser_focus_offset = 0.1
        self.disperser_pointing_offsets = np.array([0.05, -0.05])

        self.user_offsets = {
            "m1": 0.0,
            "m2": 0.0,
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "u": 0.0,
            "v": 0.0,
        }

        self.expected_corrections = {
            "m1": False,
            "m2": False,
            "hexapod": False,
            "focus": False,
            "moveWhileExposing": False,
            "atspectrograph": False,
        }

        return super().setUp()

    async def test_standard_state_transitions(self) -> None:
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
        async with self.make_csc():
            await self.check_standard_state_transitions(
                enabled_commands=(
                    "applyCorrection",
                    "applyFocusOffset",
                    "setCorrectionModelOffsets",
                    "offset",
                    "resetOffset",
                    "enableCorrection",
                    "disableCorrection",
                    "setWavelength",
                )
            )

    async def test_bin_script(self) -> None:
        await self.check_bin_script(
            name="ATAOS",
            index=None,
            exe_name="run_ataos_csc",
        )

    async def test_enable_twice(self) -> None:
        async with self.make_csc():
            await salobj.set_summary_state(self.remote, salobj.State.ENABLED)
            await salobj.set_summary_state(self.remote, salobj.State.STANDBY)
            await salobj.set_summary_state(self.remote, salobj.State.ENABLED)

    async def test_apply_correction_fail_if_corrections_enabled(self) -> None:
        """Test applyCorrection command.

        This commands applies the corrections for the current telescope
        position. It only works when the correction loop is not enabled.
        """

        async with self.make_csc(), self.mock_auxtel(), self.enable_csc():

            try:
                self.log.debug("Enabling ATAOS")
                await salobj.set_summary_state(self.remote, salobj.State.ENABLED)
                self.csc.cmd_timeout = 2.0  # 1s will timeout occasionally

                await self.remote.cmd_enableCorrection.set_start(
                    m1=True, timeout=STD_TIMEOUT
                )
            except Exception as exception:
                self.log.error("Failed to setup test.")
                raise exception

            with self.assertRaises(salobj.AckError):
                await self.remote.cmd_applyCorrection.start(timeout=STD_TIMEOUT)

    async def test_apply_correction_no_correction_while_exposing(self) -> None:
        """Test applyCorrection command.

        This commands applies the corrections for the current telescope
        position. It only works when the correction loop is not enabled.
        """

        async with self.make_csc(), self.mock_auxtel(), self.enable_csc():

            await self.camera.evt_shutterDetailedState.set_write(
                substate=ataos.ShutterState.OPEN,
            )

            await asyncio.sleep(self.csc.heartbeat_interval)

            await self.remote.cmd_applyCorrection.start(timeout=STD_TIMEOUT)

            await self.assert_apply_correction(
                while_exposing=True,
                elevation=self._telescope_elevation,
                azimuth=self._telescope_azimuth,
            )

    async def test_apply_correction(self) -> None:
        """Test applyCorrection command.

        This commands applies the corrections for the current telescope
        position. It only works when the correction loop is not enabled.
        """

        async with self.make_csc(), self.mock_auxtel(), self.enable_csc():

            elevation = self._telescope_elevation
            azimuth = self._telescope_azimuth

            await self.remote.cmd_applyCorrection.start(timeout=STD_TIMEOUT)

            await self.assert_apply_correction(
                while_exposing=False,
                elevation=elevation,
                azimuth=azimuth,
            )

    async def test_offset_fail_corrections_disabled(self) -> None:

        async with self.make_csc(), self.mock_auxtel(), self.enable_csc():

            offset = {
                "m1": 1.0,
                "m2": 1.0,
                "x": 1.0,
                "y": 1.0,
                "z": 1.0,
                "u": 1.0,
                "v": 1.0,
            }

            with self.assertRaises(salobj.AckError):
                await self.remote.cmd_offset.set_start(**offset, timeout=STD_TIMEOUT)

    async def test_offset(self) -> None:

        async with self.make_csc(), self.mock_auxtel(), self.enable_csc():

            await self.remote.cmd_enableCorrection.set_start(
                m1=True,
                m2=True,
                hexapod=True,
                moveWhileExposing=False,
                timeout=STD_TIMEOUT,
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

            self.remote.evt_correctionOffsets.flush()

            await self.remote.cmd_offset.set_start(**offset, timeout=STD_TIMEOUT)

            await self.assert_next_sample(
                topic=self.remote.evt_correctionOffsets,
                **offset,
            )

    async def test_spectrograph_offsets(self) -> None:

        async with self.make_csc(), self.mock_auxtel(), self.enable_csc():

            self.flush_spectrograph_samples()

            await self.remote.cmd_enableCorrection.set_start(
                atspectrograph=True,
                hexapod=True,
                timeout=STD_TIMEOUT,
            )

            await self.assert_correction_atspectrograph()

    async def test_spectrograph_offsets_with_user_offset(self) -> None:

        async with self.make_csc(), self.mock_auxtel(), self.enable_csc():

            self.flush_spectrograph_samples()

            await self.remote.cmd_enableCorrection.set_start(
                atspectrograph=True,
                hexapod=True,
                timeout=STD_TIMEOUT,
            )

            await self.apply_user_offsets(
                offset={
                    "x": 1.3,
                    "y": 1.4,
                    "z": 1.5,
                    "u": 1.6,
                    "v": 1.7,
                }
            )
            await self.assert_spectrograph_offsets_with_user_offset()

    async def test_spectrograph_offsets_change_filter(self) -> None:

        async with self.make_csc(), self.mock_auxtel(), self.enable_csc():

            await self.remote.cmd_enableCorrection.set_start(
                atspectrograph=True,
                hexapod=True,
                timeout=STD_TIMEOUT,
            )

            self.flush_spectrograph_samples()

            self.filter_name = "test_filt2"
            self.filter_focus_offset = 0.01
            self.filter_central_wavelength = 700.0
            self.filter_pointing_offsets = np.array([0.2, -0.2])

            await self._atspectrograph_reported_filter_position()

            await self.assert_correction_atspectrograph()

    async def test_spectrograph_offsets_republished_filter(self) -> None:

        async with self.make_csc(), self.mock_auxtel(), self.enable_csc():

            await self.remote.cmd_enableCorrection.set_start(
                atspectrograph=True,
                hexapod=True,
                timeout=STD_TIMEOUT,
            )

            self.flush_spectrograph_samples()

            offset_call_count = self.atptg.cmd_poriginOffset.callback.call_count

            await self._atspectrograph_reported_filter_position(force=True)

            await self.assert_spectrograph_offsets_spectrograph_republished(
                offset_call_count
            )

    async def test_spectrograph_offsets_change_disperser(self) -> None:

        async with self.make_csc(), self.mock_auxtel(), self.enable_csc():

            await self.remote.cmd_enableCorrection.set_start(
                atspectrograph=True,
                hexapod=True,
                timeout=STD_TIMEOUT,
            )

            self.flush_spectrograph_samples()

            self.disperser_name = "test_disp2"
            self.disperser_focus_offset = 0.3
            self.disperser_pointing_offsets = np.array([0.13, -0.13])

            await self._atspectrograph_reported_disperser_position()

            await self.assert_correction_atspectrograph()

    async def test_spectrograph_offsets_republished_disperser(self) -> None:

        async with self.make_csc(), self.mock_auxtel(), self.enable_csc():

            await self.remote.cmd_enableCorrection.set_start(
                atspectrograph=True,
                hexapod=True,
                timeout=STD_TIMEOUT,
            )

            self.flush_spectrograph_samples()

            offset_call_count = self.atptg.cmd_poriginOffset.callback.call_count

            await self._atspectrograph_reported_disperser_position(force=True)

            await self.assert_spectrograph_offsets_spectrograph_republished(
                offset_call_count
            )

    async def test_enable_corrections_fails_no_correction_set(self) -> None:
        async with self.make_csc(), self.mock_auxtel(), self.enable_csc():
            with self.assertRaises(salobj.AckError):
                await self.remote.cmd_enableCorrection.set_start(timeout=STD_TIMEOUT)

    async def test_enable_corrections_m1(self) -> None:
        async with self.make_csc(), self.mock_auxtel(), self.enable_csc():

            await self.assert_enable_corrections(m1=True)

    async def test_enable_corrections_m2(self) -> None:
        async with self.make_csc(
            config_dir=TEST_CONFIG_DIR
        ), self.mock_auxtel(), self.enable_csc():

            await self.assert_enable_corrections(m2=True)

    async def test_enable_corrections_hexapod(self) -> None:
        async with self.make_csc(), self.mock_auxtel(), self.enable_csc():

            await self.assert_enable_corrections(hexapod=True)

    async def test_enable_corrections_atspectrograph(self) -> None:
        async with self.make_csc(), self.mock_auxtel(), self.enable_csc():

            self.flush_spectrograph_samples()

            await self.assert_enable_corrections(atspectrograph=True)

    async def assert_enable_corrections(self, **kwargs: typing.Any) -> None:

        self.expected_corrections.update(kwargs)

        self.remote.evt_correctionEnabled.flush()

        await self.remote.cmd_enableCorrection.set_start(
            **self.expected_corrections, timeout=STD_TIMEOUT
        )

        await self.assert_next_sample(
            topic=self.remote.evt_correctionEnabled,
            **self.expected_corrections,
        )

        for correction in self.expected_corrections:
            if self.expected_corrections[correction]:
                await self.assert_correction(correction)

    async def apply_user_offsets(self, offset: typing.Dict[str, float]) -> None:

        self.user_offsets.update(offset)

        await self.remote.cmd_offset.set_start(**offset, timeout=STD_TIMEOUT)

    async def assert_apply_correction(
        self,
        while_exposing: bool,
        elevation: float,
        azimuth: float,
    ) -> None:

        self.pnematics.cmd_m1SetPressure.callback.assert_awaited()
        self.pnematics.cmd_m2SetPressure.callback.assert_awaited()

        if while_exposing:
            self.hexapod.cmd_moveToPosition.callback.assert_not_awaited()
            self.atptg.cmd_poriginOffset.callback.assert_not_awaited()
        else:
            self.hexapod.cmd_moveToPosition.callback.assert_awaited()
            self.atptg.cmd_poriginOffset.callback.assert_awaited()

        self.remote.evt_detailedState.callback.assert_awaited()

        await self.assert_next_sample(
            topic=self.remote.evt_m1CorrectionStarted,
            elevation=elevation,
            azimuth=azimuth,
        )
        await self.assert_next_sample(
            topic=self.remote.evt_m2CorrectionStarted,
            elevation=elevation,
            azimuth=azimuth,
        )
        await self.assert_next_sample(
            topic=self.remote.evt_m1CorrectionCompleted,
            elevation=elevation,
            azimuth=azimuth,
        )
        await self.assert_next_sample(
            topic=self.remote.evt_m2CorrectionCompleted,
            elevation=elevation,
            azimuth=azimuth,
        )

        if while_exposing:
            events = await asyncio.gather(
                self.remote.evt_hexapodCorrectionStarted.next(
                    flush=False, timeout=STD_TIMEOUT
                ),
                self.remote.evt_hexapodCorrectionCompleted.next(
                    flush=False, timeout=STD_TIMEOUT
                ),
                self.remote.evt_atspectrographCorrectionStarted.next(
                    flush=False, timeout=STD_TIMEOUT
                ),
                self.remote.evt_atspectrographCorrectionCompleted.next(
                    flush=False, timeout=STD_TIMEOUT
                ),
                return_exceptions=True,
            )
            assert all([isinstance(event, asyncio.TimeoutError) for event in events])

        else:
            await self.assert_next_sample(
                topic=self.remote.evt_hexapodCorrectionStarted,
                elevation=elevation,
                azimuth=azimuth,
            )
            await self.assert_next_sample(
                topic=self.remote.evt_hexapodCorrectionCompleted,
                elevation=elevation,
                azimuth=azimuth,
            )
            expected_atspectrograph_corrections = {
                "focusOffset": pytest.approx(
                    self.filter_focus_offset + self.disperser_focus_offset
                ),
                "pointingOffsets": [
                    pytest.approx(fp + dp)
                    for fp, dp in zip(
                        self.filter_pointing_offsets, self.disperser_pointing_offsets
                    )
                ],
            }
            await self.assert_next_sample(
                topic=self.remote.evt_atspectrographCorrectionStarted,
                **expected_atspectrograph_corrections,
            )
            await self.assert_next_sample(
                topic=self.remote.evt_atspectrographCorrectionCompleted,
                **expected_atspectrograph_corrections,
            )

        # Check how many times the detailed state bit was flipped
        # Should be a transition per component (2 flips), except
        # the spectrograph calls the hexapod, so it's actually
        # 4*2+2 = 10 detailed state transitions.
        # hexapod nor spectrograph function while exposing, so only
        # the mirrors should flip.
        assert len(self.remote.evt_detailedState.callback.call_args_list) == (
            10 if not while_exposing else 4
        ), ("%s" % self.remote.evt_detailedState.callback.call_args_list)

    def flush_spectrograph_samples(self) -> None:
        self.remote.evt_pointingOffsetSummary.flush()
        self.remote.evt_atspectrographCorrectionCompleted.flush()
        self.remote.evt_atspectrographCorrectionStarted.flush()
        self.remote.evt_correctionOffsets.flush()
        self.remote.evt_focusOffsetSummary.flush()

    async def assert_correction(self, correction: str) -> None:
        await getattr(self, f"assert_correction_{correction}")()

    async def assert_correction_m1(self) -> None:

        self.pnematics.cmd_m1SetPressure.callback.assert_called()
        self.pnematics.cmd_m1OpenAirValve.callback.assert_called()

        await self.assert_next_sample(topic=self.remote.evt_m1CorrectionStarted)
        await self.assert_next_sample(topic=self.remote.evt_m1CorrectionCompleted)

    async def assert_correction_m2(self) -> None:

        self.pnematics.cmd_m2SetPressure.callback.assert_called()
        self.pnematics.cmd_m2OpenAirValve.callback.assert_called()

        await self.assert_next_sample(topic=self.remote.evt_m2CorrectionStarted)
        await self.assert_next_sample(topic=self.remote.evt_m2CorrectionCompleted)

    async def assert_correction_hexapod(self) -> None:

        self.hexapod.cmd_moveToPosition.callback.assert_called()

        await self.assert_next_sample(topic=self.remote.evt_hexapodCorrectionStarted)
        await self.assert_next_sample(topic=self.remote.evt_hexapodCorrectionCompleted)

    async def assert_correction_atspectrograph(self) -> None:

        self.atptg.cmd_poriginOffset.callback.assert_called()

        await self.assert_spectrograph_offsets_expected_values(
            expected_values=dict(
                pointing_offset_summary=dict(
                    filter=self.filter_pointing_offsets,
                    disperser=self.disperser_pointing_offsets,
                    total=np.array(
                        [
                            filter_pointing_offset + disperser_pointing_offset
                            for filter_pointing_offset, disperser_pointing_offset in zip(
                                self.filter_pointing_offsets,
                                self.disperser_pointing_offsets,
                            )
                        ]
                    ),
                ),
                correction_offsets=dict(
                    z=pytest.approx(
                        self.filter_focus_offset + self.disperser_focus_offset
                    ),
                ),
                focus_offset_summary=dict(
                    total=pytest.approx(
                        self.disperser_focus_offset + self.filter_focus_offset
                    ),
                ),
            )
        )

        await self.assert_next_sample(
            topic=self.remote.evt_atspectrographCorrectionStarted,
        )

        await self.assert_next_sample(
            topic=self.remote.evt_atspectrographCorrectionCompleted,
        )

    async def assert_spectrograph_offsets_with_user_offset(self) -> None:

        await self.assert_correction_atspectrograph()

        correction_offsets = dict(
            [(k, pytest.approx(self.user_offsets[k])) for k in self.user_offsets]
        )
        correction_offsets["z"] = pytest.approx(
            self.user_offsets["z"]
            + self.disperser_focus_offset
            + self.filter_focus_offset
        )

        await self.assert_spectrograph_offsets_expected_values(
            expected_values=dict(
                correction_offsets=correction_offsets,
                focus_offset_summary=dict(
                    total=pytest.approx(correction_offsets["z"]),
                    userApplied=pytest.approx(self.user_offsets["z"]),
                    filter=pytest.approx(self.filter_focus_offset),
                    disperser=pytest.approx(self.disperser_focus_offset),
                ),
            ),
        )

    async def assert_spectrograph_offsets_spectrograph_republished(
        self, offset_call_count: int
    ) -> None:

        await self.assert_correction_atspectrograph()

        assert offset_call_count == self.atptg.cmd_poriginOffset.callback.call_count

    async def assert_spectrograph_offsets_expected_values(
        self, expected_values: typing.Any
    ) -> None:

        if "pointing_offset_summary" in expected_values:
            pointing_offset_summary = await self.assert_next_sample(
                topic=self.remote.evt_pointingOffsetSummary,
            )

            for attr in expected_values["pointing_offset_summary"]:
                assert getattr(pointing_offset_summary, attr) == pytest.approx(
                    expected_values["pointing_offset_summary"][attr]
                )

        if "correction_offsets" in expected_values:
            await self.assert_next_sample(
                topic=self.remote.evt_correctionOffsets,
                **expected_values["correction_offsets"],
            )

        if "focus_offset_summary" in expected_values:
            await self.assert_next_sample(
                topic=self.remote.evt_focusOffsetSummary,
                **expected_values["focus_offset_summary"],
            )

    @contextlib.asynccontextmanager
    async def enable_csc(self) -> typing.AsyncGenerator[None, None]:
        try:
            self.log.debug("Enabling ATAOS")
            self.remote.evt_detailedState.callback = unittest.mock.AsyncMock()
            await salobj.set_summary_state(self.remote, salobj.State.ENABLED)
            yield
            await salobj.set_summary_state(self.remote, salobj.State.STANDBY)
        except Exception:
            self.log.exception("Failed to enable ATAOS")
            raise

    @contextlib.asynccontextmanager
    async def mock_auxtel(self) -> typing.AsyncGenerator[None, None]:

        try:

            async with salobj.Controller("ATMCS") as self.atmcs, salobj.Controller(
                "ATPtg"
            ) as self.atptg, salobj.Controller(
                "ATPneumatics"
            ) as self.pnematics, salobj.Controller(
                "ATHexapod"
            ) as self.hexapod, salobj.Controller(
                "ATCamera"
            ) as self.camera, salobj.Controller(
                "ATSpectrograph"
            ) as self.atspectrograph:

                self.set_pneumatics_callbacks()
                self.set_atptg_callbacks()
                self.set_athexapod_callbacks()

                await self.publish_pneumatics_initial_data()
                await self.publish_atspectrograph_initial_data()

                self.running = True

                mount_telemetry_task = asyncio.create_task(
                    self.publish_mount_encoders()
                )
                yield
                self.running = False
                try:
                    await asyncio.wait_for(mount_telemetry_task, timeout=STD_TIMEOUT)
                except asyncio.TimeoutError:
                    mount_telemetry_task.cancel()

        except Exception as exception:
            raise exception

    def set_pneumatics_callbacks(self) -> None:
        self.pnematics.cmd_m1SetPressure.callback = unittest.mock.AsyncMock(
            wraps=self.m1_set_pressure_callback
        )
        self.pnematics.cmd_m2SetPressure.callback = unittest.mock.AsyncMock(
            wraps=self.m2_set_pressure_callback
        )
        self.pnematics.cmd_m1OpenAirValve.callback = unittest.mock.AsyncMock(
            wraps=self.m1_open_callback
        )
        self.pnematics.cmd_m2OpenAirValve.callback = unittest.mock.AsyncMock(
            wraps=self.m2_open_callback
        )
        self.pnematics.cmd_m1CloseAirValve.callback = unittest.mock.AsyncMock(
            wraps=self.m1_close_callback
        )
        self.pnematics.cmd_m2CloseAirValve.callback = unittest.mock.AsyncMock(
            wraps=self.m2_close_callback
        )

    def set_atptg_callbacks(self) -> None:
        self.atptg.cmd_poriginOffset.callback = unittest.mock.AsyncMock(
            wraps=self.mount_offset_callback
        )

    def set_athexapod_callbacks(self) -> None:
        self.hexapod.cmd_moveToPosition.callback = unittest.mock.AsyncMock(
            wraps=self.hexapod_move_callback
        )

    async def publish_pneumatics_initial_data(self) -> None:
        await self.pnematics.evt_summaryState.set_write(
            summaryState=salobj.State.ENABLED
        )
        await self.pnematics.evt_mainValveState.set_write(
            state=ATPneumatics.AirValveState.OPENED
        )
        await self.pnematics.evt_instrumentState.set_write(
            state=ATPneumatics.AirValveState.OPENED
        )
        await self.pnematics.evt_m1State.set_write(
            state=ATPneumatics.AirValveState.CLOSED
        )
        await self.pnematics.evt_m2State.set_write(
            state=ATPneumatics.AirValveState.CLOSED
        )

    async def publish_atspectrograph_initial_data(self) -> None:

        await self.atspectrograph.evt_summaryState.set_write(
            summaryState=salobj.State.ENABLED
        )

        await self._atspectrograph_reported_filter_position()

        await self._atspectrograph_reported_disperser_position()

    async def _atspectrograph_reported_filter_position(
        self, force: bool = False
    ) -> None:

        await self.atspectrograph.evt_reportedFilterPosition.set_write(
            name=self.filter_name,
            centralWavelength=self.filter_central_wavelength,
            focusOffset=self.filter_focus_offset,
            pointingOffsets=self.filter_pointing_offsets,
            force_output=force,
        )

    async def _atspectrograph_reported_disperser_position(
        self, force: bool = False
    ) -> None:

        await self.atspectrograph.evt_reportedDisperserPosition.set_write(
            name=self.disperser_name,
            focusOffset=self.disperser_focus_offset,
            pointingOffsets=self.disperser_pointing_offsets,
            force_output=force,
        )

    async def hexapod_move_callback(self, data: typing.Any) -> None:
        await self.hexapod.evt_positionUpdate.set_write(
            positionX=data.x,
            positionY=data.y,
            positionZ=data.z,
            positionU=data.u,
            positionV=data.v,
        )

    async def m1_set_pressure_callback(self, data: typing.Any) -> None:
        await self.pnematics.tel_m1AirPressure.set_write(pressure=data.pressure)

    async def m2_set_pressure_callback(self, data: typing.Any) -> None:
        await self.pnematics.tel_m2AirPressure.set_write(pressure=data.pressure)

    async def m1_open_callback(self, data: typing.Any) -> None:
        await self.pnematics.evt_m1State.set_write(
            state=ATPneumatics.AirValveState.OPENED
        )

    async def m2_open_callback(self, data: typing.Any) -> None:
        await self.pnematics.evt_m2State.set_write(
            state=ATPneumatics.AirValveState.OPENED
        )

    async def m1_close_callback(self, data: typing.Any) -> None:
        await self.pnematics.evt_m1State.set_write(
            state=ATPneumatics.AirValveState.CLOSED
        )

    async def m2_close_callback(self, data: typing.Any) -> None:
        await self.pnematics.evt_m2State.set_write(
            state=ATPneumatics.AirValveState.CLOSED
        )

    async def mount_offset_callback(self, data: typing.Any) -> None:
        # just assume 0.1 degree offsets in both axis
        await self.atmcs.evt_allAxesInPosition.set_write(inPosition=False)
        self._telescope_azimuth += 0.1
        self._telescope_elevation += 0.1
        await asyncio.sleep(salobj.base_csc.HEARTBEAT_INTERVAL)
        await self.atmcs.evt_allAxesInPosition.set_write(inPosition=True)

    async def publish_mount_encoders(self) -> None:
        """Publish telescope position data.

        Nasmyth values are just equal to azimuth
        """

        while self.running:
            _azimuth = np.zeros(100) + self._telescope_azimuth
            # make sure it is never zero because np.random.uniform is
            # [min, max)
            _elevation = np.zeros(100) + self._telescope_elevation

            await self.atmcs.tel_mount_AzEl_Encoders.set_write(
                azimuthCalculatedAngle=_azimuth, elevationCalculatedAngle=_elevation
            )
            # Assume nasmyth is the same as the azimuth
            await self.atmcs.tel_mount_Nasmyth_Encoders.set_write(
                nasmyth1CalculatedAngle=_azimuth,
                nasmyth2CalculatedAngle=_azimuth,
            )
            await asyncio.sleep(salobj.base_csc.HEARTBEAT_INTERVAL)


if __name__ == "__main__":
    unittest.main()
