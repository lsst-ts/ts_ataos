
import unittest
import numpy as np

from lsst.ts.ataos import Model


class TestModel(unittest.TestCase):

    def test_corrections(self):

        model = Model()

        # test default configuration

        test_altitudes = np.linspace(0., 90., 5)
        test_azimuth = np.linspace(0., 90., 5)
        test_temperatures = np.linspace(-1., 30., 3)

        for alt in test_altitudes:
            for az in test_azimuth:
                for temp in test_temperatures:
                    # By default, all corrections should be zero.
                    with self.subTest(correction="m1", alt=alt, az=az, temp=temp):
                        self.assertEqual(0.,
                                         model.get_correction_m1(alt, az, temp))
                    with self.subTest(correction="m2", alt=alt, az=az, temp=temp):
                        self.assertEqual(0.,
                                         model.get_correction_m2(alt, az, temp))
                    with self.subTest(correction="hexapod", alt=alt, az=az, temp=temp):
                        for corr in model.get_correction_hexapod(alt, az, temp):
                            self.assertEqual(0.,
                                             corr)

        test_config = {'m1': [np.random.rand()],
                       'm2': [np.random.rand()],
                       'hexapod_x': [np.random.rand()],
                       'hexapod_y': [np.random.rand()],
                       'hexapod_z': [np.random.rand()],
                       'hexapod_u': [np.random.rand()],
                       'hexapod_v': [np.random.rand()],
                       }

        model.config = test_config

        # correction is a polynome so all values should be equal to the
        # values on test_config
        axis = 'xyzuv'

        for alt in test_altitudes:
            for az in test_azimuth:
                for temp in test_temperatures:
                    with self.subTest(correction="m1", alt=alt, az=az, temp=temp):
                        self.assertEqual(test_config["m1"][0],
                                         model.get_correction_m1(alt, az, temp))
                    with self.subTest(correction="m2", alt=alt, az=az, temp=temp):
                        self.assertEqual(test_config["m2"][0],
                                         model.get_correction_m2(alt, az, temp))
                    with self.subTest(correction="hexapod", alt=alt, az=az, temp=temp):
                        hexapod = model.get_correction_hexapod(alt, az, temp)
                        for i in range(len(axis)):
                            self.assertEqual(test_config[f"hexapod_{axis[i]}"][0],
                                             hexapod[i])


if __name__ == '__main__':

    unittest.main()
