from pathlib import Path
import os
import shutil
import subprocess
from types import SimpleNamespace
import tempfile
from contextlib import ExitStack
import pprint

import benchr
from benchr import (
    Benchmark,
    Config,
    DryExecutor,
    DefaultExecutor,
    Parameters,
    Suite,
    config_to_runs,
)

CWD = Path("/home/rihafilip/code/r/rcp/rcp")
# TODO:
# Assuming this is in `rcp` subfolder of PRL_PRG/rcp
# CWD = Path(__file__).parent


def check_microbenchmark(Rscript: Path):
    benchr.run_cmd(
        [
            Rscript,
            "-e",
            """if (!requireNamespace("microbenchmark", quietly=TRUE)) quit(status=1)""",
        ]
    )


def main():
    with ExitStack() as estack:
        params = benchr.parse_params(
            RSH_HOME=CWD / ".." / "external" / "rsh" / "client" / "rsh",
            R_HOME=CWD / ".." / "external" / "rsh" / "external" / "R",
            bench_opts="--rcp",
            filter="",
            parallel=os.cpu_count(),
            runs=1,
            output=None,
        )

        RSH_HOME = params.RSH_HOME.resolve()
        R_HOME = params.R_HOME.resolve()
        bench_opts = params.bench_opts.split()
        filter = params.filter
        parallel = params.parallel
        runs = params.runs
        output = (
            params.output
            if params.output is not None
            else estack.enter_context(tempfile.TemporaryDirectory())
        )

        bench_dir = RSH_HOME / "inst" / "benchmarks"
        harness_bin = RSH_HOME / "inst" / "benchmarks" / "harness.R"

        R = R_HOME / "bin" / "R"
        Rscript = R_HOME / "bin" / "Rscript"

        benchmarks = [
            Benchmark(path.stem, path)
            for path in bench_dir.rglob(f"*{filter}*.R")
            # Top level has main program and harness -> we want benchmarks
            if path.parent != bench_dir
        ]

        time = shutil.which("time")
        if time is None:
            raise ValueError("time utility is not available")

        RCPSuite = Suite(
            name="RCPSuite",
            benchmarks=benchmarks,
            working_directory=CWD,
            command=lambda _, benchmark: (
                [time, "-v"]
                + [str(R), "--slave", "--no-restore"]
                + ["-f", str(harness_bin), "--args"]
                + ["--output-dir", output]
                + ["--runs", str(runs)]
                + bench_opts
                + [str(benchmark.data.with_suffix(""))]
            ),
        )

        runs = config_to_runs(Config(RCPSuite), params)

        check_microbenchmark(Rscript)

        # TODO: Executor
        # executor = DryExecutor()
        with DefaultExecutor(benchr.RebenchParser(), benchr.CsvFormatter()) as executor:
            executor.execute_all(runs)


if __name__ == "__main__":
    main()
