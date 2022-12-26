# import sys
# import ast
# from refactor import Session, Rule, common
# from refactor.actions import Replace
# from refactor.actions import InsertAfter
# from pathlib import Path
#
import tracemalloc
import time
import logging
from TPUMigrator import TPUMigrator

def tracing_start():
    tracemalloc.stop()
    print("nTracing Status : ", tracemalloc.is_tracing())
    tracemalloc.start()
    print("Tracing Status : ", tracemalloc.is_tracing())
def tracing_mem():
    first_size, first_peak = tracemalloc.get_traced_memory()
    peak = first_peak/(1024*1024)
    print("Peak Size in MB - ", peak)

if __name__ == "__main__":
    tracing_start()
    start = time.time()
    migrator = TPUMigrator("misc.py", level = logging.INFO)
    migrator.execute(comments=True, reformatter=True, optimize = True, genTestCases = False, PYNGUINPATH="")
    end = time.time()
    print("time elapsed {} milli seconds".format((end - start) * 1000))
    tracing_mem()
