#!/usr/bin/env python

"""
A basic script to emulate the athexapod for integration tests.
"""

import asyncio

from lsst.ts.salobj import Controller
import SALPY_ATHexapod


async def fake_move_to_position(id_data):
    print("**********")
    for axis in f'xyzuvw':
        print(f"Moving {axis}: {getattr(id_data.data, axis)}")
        await asyncio.sleep(0.5)
    print("**********")


if __name__ == "__main__":

    print("Starting Controller")
    athex = Controller(SALPY_ATHexapod)
    print("Adding callback")
    athex.cmd_moveToPosition.callback = fake_move_to_position
    print("Run Controller forever")
    asyncio.get_event_loop().run_forever()
