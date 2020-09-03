#!/usr/bin/env python

import asyncio

from lsst.ts.ataos import ataos_csc

asyncio.run(ataos_csc.ATAOS.amain(index=None))
