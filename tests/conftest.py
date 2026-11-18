import os

# torch + scikit-learn/pandas load different OpenMP runtimes on macOS; a single thread avoids a segfault
# in the CPU LSTM when both are used in one test process.
os.environ.setdefault("OMP_NUM_THREADS", "1")
