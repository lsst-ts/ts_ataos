import typing
import unittest
import numpy as np
import logging
from lsst.ts.ataos import Model

logger = logging.getLogger(__name__)


class TestModel(unittest.TestCase):
    def setUp(self) -> None:
        self.model = Model(logger)

        # test default configuration

        self.test_altitudes = np.linspace(0.0, 90.0, 5)
        self.test_azimuth = np.linspace(0.0, 90.0, 5)
        self.test_temperatures = np.linspace(-1.0, 30.0, 3)
        self.test_wavelengths = np.linspace(320.0, 1100.0, 10)  # min, max, steps

    def test_corrections_default_values(self) -> None:

        self.assert_ataos_corrections(self.get_zero_test_config())

    def test_corrections_random_values(self) -> None:

        test_config = self.get_random_value_test_config()

        self.set_model_test_config_values(test_config)

        self.assert_ataos_corrections(test_config)

        # Note below that the central wavelength for the telescope
        # without any filter is 700nm
        for wave in self.test_wavelengths:
            logger.debug(f'test_config value is {test_config["chromatic_dependence"]}')

            with self.subTest(correction="chromatic_dependence", wave=wave):
                self.assertEqual(
                    test_config["chromatic_dependence"][0],
                    self.model.get_correction_chromatic(wave - 700),
                )

    def test_corrections_hexapod_sensitivity_matrix_x(self) -> None:

        # This sensitivity matrix will make the correction in x propaget to all
        # other axis, while nullifying the correction on the axis itself.
        # For example is there is 0 correction in x, and 10 correction in y,
        # the resulting correction will be zero in all axis. On the other hand
        # if the correction is 10 in x it will be 10 in all other axis,
        # regardless of the correction on those axis.
        hexapod_sensitivity_matrix_propagete_axis = np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # x
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # y
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # z
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # u
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # v
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # w
            ]
        )

        test_config = self.get_random_value_test_config()

        self.set_model_test_config_values(test_config)

        check_values = self.copy_values_from_axis(test_config, "x")

        self.run_test_corrections_hexapod_sensitivity_matrix(
            hexapod_sensitivity_matrix_propagete_axis, test_config, check_values
        )

    def test_corrections_hexapod_sensitivity_matrix_y(self) -> None:

        hexapod_sensitivity_matrix_propagete_axis = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # x
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # y
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # z
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # u
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # v
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # w
            ]
        )

        test_config = self.get_random_value_test_config()

        self.set_model_test_config_values(test_config)

        check_values = self.copy_values_from_axis(test_config, "y")

        self.run_test_corrections_hexapod_sensitivity_matrix(
            hexapod_sensitivity_matrix_propagete_axis, test_config, check_values
        )

    def test_corrections_hexapod_sensitivity_matrix_z(self) -> None:

        hexapod_sensitivity_matrix_propagete_axis = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # x
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # y
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # z
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # u
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # v
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # w
            ]
        )

        test_config = self.get_random_value_test_config()

        self.set_model_test_config_values(test_config)

        check_values = self.copy_values_from_axis(test_config, "z")

        self.run_test_corrections_hexapod_sensitivity_matrix(
            hexapod_sensitivity_matrix_propagete_axis, test_config, check_values
        )

    def test_get_lut_elevation(self) -> None:

        limits = [20.0, 80.0]

        for inclination in [0.0, 5.0, 10.0, 15]:
            self.assertEqual(self.model.get_lut_elevation(inclination, limits), 20.0)

        for inclination in [20.0, 40.0, 60.0, 80.0]:
            self.assertEqual(
                self.model.get_lut_elevation(inclination, limits), inclination
            )

        for inclination in [82.5, 85.0, 87.5, 90.0]:
            self.assertEqual(self.model.get_lut_elevation(inclination, limits), 80.0)

    def test_set_hexapod_sensitivity_matrix_as_list(self) -> None:

        new_hexapod_sensitivity_matrix = [
            [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # x
            [0.0, -1.0, 0.0, 0.0, 0.0, 0.0],  # y
            [0.0, 0.0, -1.0, 0.0, 0.0, 0.0],  # z
            [0.0, 0.0, 0.0, -1.0, 0.0, 0.0],  # u
            [0.0, 0.0, 0.0, 0.0, -1.0, 0.0],  # v
            [0.0, 0.0, 0.0, 0.0, 0.0, -1.0],  # w
        ]

        self.model.hexapod_sensitivity_matrix = new_hexapod_sensitivity_matrix

        for line_original, line_set in zip(
            new_hexapod_sensitivity_matrix,
            self.model.hexapod_sensitivity_matrix,
        ):
            for value_original, value_set in zip(line_original, line_set):
                self.assertEqual(value_original, value_set)

    def test_set_hexapod_sensitivity_matrix_wrong_size(self) -> None:

        bad_hexapod_sensitivity_matrix_5x6 = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # x
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # y
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # z
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # u
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # v
            ]
        )

        bad_hexapod_sensitivity_matrix_6x5 = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],  # x
                [0.0, 1.0, 0.0, 0.0, 0.0],  # y
                [0.0, 0.0, 1.0, 0.0, 0.0],  # z
                [0.0, 0.0, 0.0, 1.0, 0.0],  # u
                [0.0, 0.0, 0.0, 0.0, 1.0],  # v
                [0.0, 0.0, 0.0, 0.0, 0.0],  # w
            ]
        )

        with self.assertRaises(RuntimeError):
            self.model.hexapod_sensitivity_matrix = bad_hexapod_sensitivity_matrix_5x6

        with self.assertRaises(RuntimeError):
            self.model.hexapod_sensitivity_matrix = bad_hexapod_sensitivity_matrix_6x5

    def test_get_correction_m1_out_of_bound_factor(self) -> None:

        correction_m1_lut = 1.0

        self.model.m1_lut_elevation_limits = [20.0, 80]

        m1_out_of_bound_factor = self.model.get_correction_m1_out_of_bound_factor(
            0.0, correction_m1_lut
        )
        self.assertEqual(m1_out_of_bound_factor, -1.0 * correction_m1_lut)

        m1_out_of_bound_factor = self.model.get_correction_m1_out_of_bound_factor(
            self.model.m1_lut_elevation_limits[0], correction_m1_lut
        )
        self.assertEqual(m1_out_of_bound_factor, 0.0)

        for elevation in np.linspace(0.0, self.model.m1_lut_elevation_limits[0]):
            m1_out_of_bound_factor = self.model.get_correction_m1_out_of_bound_factor(
                elevation, correction_m1_lut
            )

            self.assertEqual(
                m1_out_of_bound_factor,
                correction_m1_lut
                * (elevation / self.model.m1_lut_elevation_limits[0] - 1.0),
            )

        for elevation in np.linspace(self.model.m1_lut_elevation_limits[1], 90.0):
            m1_out_of_bound_factor = self.model.get_correction_m1_out_of_bound_factor(
                elevation, correction_m1_lut
            )

            self.assertEqual(m1_out_of_bound_factor, 0.0)

    def copy_values_from_axis(
        self,
        test_config: typing.Dict[str, typing.Any],
        copy_axis: str,
    ) -> typing.Dict[str, typing.Any]:
        check_values = dict(**test_config)

        for index, axis in enumerate("xyzuv"):
            check_values[f"hexapod_{axis}"] = test_config[f"hexapod_{copy_axis}"]

        return check_values

    def run_test_corrections_hexapod_sensitivity_matrix(
        self,
        test_hexapod_sensitivity_matrix: np.ndarray,
        test_config: typing.Dict[str, typing.Any],
        check_values: typing.Dict[str, typing.Any],
    ) -> None:

        self.model.hexapod_sensitivity_matrix = test_hexapod_sensitivity_matrix

        self.set_model_test_config_values(test_config)

        self.assert_ataos_corrections(check_values)

    def assert_ataos_corrections(
        self, corrections: typing.Dict[str, typing.Any]
    ) -> None:

        # correction is a polynomial so all values should be equal to the
        # values on test_config
        axis = "xyzuv"

        for alt in self.test_altitudes:
            for az in self.test_azimuth:
                for temp in self.test_temperatures:
                    self.assertEqual(
                        corrections["m1"][0],
                        self.model.get_correction_m1(alt, az, temp),
                    )
                    self.assertEqual(
                        corrections["m2"][0],
                        self.model.get_correction_m2(alt, az, temp),
                    )
                    hexapod = self.model.get_correction_hexapod(alt, az, temp)
                    for i in range(len(axis)):
                        self.assertEqual(
                            corrections[f"hexapod_{axis[i]}"][0],
                            hexapod[i],
                            f"Failed-axis: {axis[i]}",
                        )

    @staticmethod
    def get_random_value_test_config() -> typing.Dict[str, typing.List[np.ndarray]]:
        return {
            "m1": [np.random.rand()],
            "m2": [np.random.rand()],
            "hexapod_x": [np.random.rand()],
            "hexapod_y": [np.random.rand()],
            "hexapod_z": [np.random.rand()],
            "hexapod_u": [np.random.rand()],
            "hexapod_v": [np.random.rand()],
            "chromatic_dependence": [np.random.rand()],
        }

    @staticmethod
    def get_zero_test_config() -> typing.Dict[str, typing.List[float]]:
        return {
            "m1": [0.0],
            "m2": [0.0],
            "hexapod_x": [0.0],
            "hexapod_y": [0.0],
            "hexapod_z": [0.0],
            "hexapod_u": [0.0],
            "hexapod_v": [0.0],
            "chromatic_dependence": [0.0],
        }

    def set_model_test_config_values(
        self, test_config: typing.Dict[str, typing.Any]
    ) -> None:
        for key in test_config:
            setattr(self.model, key, test_config[key])


if __name__ == "__main__":

    unittest.main()
