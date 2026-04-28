from pathlib import Path
import runpy
import time


start_time = time.perf_counter()
runpy.run_path(str(Path(__file__).with_name(
    "jax_code_fixed_itr2.py")), run_name="__main__")
end_time = time.perf_counter()

print(f"Execution time: {end_time - start_time:.4f} seconds")
