#!/usr/bin/env python
"""
Parallel Hello World
"""

from mpi4py import MPI
import sys
import time

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()

sys.stdout.write(f"Hello, World! I am process {rank} of {size} on {name}.\n")

time.sleep(10)

sys.stdout.write(f"Goodbye ! I am process {rank} of {size} on {name}.\n")
