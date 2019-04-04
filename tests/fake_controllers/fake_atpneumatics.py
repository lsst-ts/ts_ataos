#!/usr/bin/env python

"""
A basic script to emulate the ATPneumatic for integration tests.
"""

import asyncio

from lsst.ts.salobj import Controller
import SALPY_ATPneumatics


async def fake_m1SetPressure(id_data):
    print("**********")
    print(f"Seeting m1 pressure to {id_data.data.pressure}")
    await asyncio.sleep(0.5)
    print("**********")


async def fake_m2SetPressure(id_data):
    print("**********")
    print(f"Seeting m2 pressure to {id_data.data.pressure}")
    await asyncio.sleep(0.5)
    print("**********")


if __name__ == "__main__":

    print("Starting Controller")
    atpne = Controller(SALPY_ATPneumatics)
    print("Adding callback")
    atpne.cmd_m1SetPressure.callback = fake_m1SetPressure
    atpne.cmd_m2SetPressure.callback = fake_m2SetPressure
    print("Run Controller forever")
    asyncio.get_event_loop().run_forever()
