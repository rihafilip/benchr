import os
import tempfile
from contextlib import ExitStack
import subprocess

from benchr import *

CWD = Path("/home/rihafilip/code/r/rcp/rcp")
# TODO:
# Assuming this is in `rcp` subfolder of PRL_PRG/rcp
# CWD = Path(__file__).parent


def check_namespace(Rscript: Path, namespace: str):
    subprocess.run(
        [
            Rscript,
            "-e",
            f"""if (!requireNamespace("{namespace}", quietly=TRUE)) quit(status=1)""",
        ],
        check=True,
    )


def rcp_main():
    with ExitStack() as estack:
        params = parse_params(
            RSH_HOME=CWD / ".." / "external" / "rsh" / "client" / "rsh",
            R_HOME=CWD / ".." / "external" / "rsh" / "external" / "R",
            path_filter="",
            parallel=os.cpu_count(),
            runs=1,
            output=None,
        )

        RSH_HOME: Path = params.RSH_HOME.resolve()
        R_HOME: Path = params.R_HOME.resolve()
        path_filter = params.path_filter
        parallel = params.parallel
        runs = params.runs
        output = (
            Path(params.output)
            if params.output is not None
            else Path(estack.enter_context(tempfile.TemporaryDirectory()))
        )

        bench_dir = RSH_HOME / "inst" / "benchmarks"
        harness_bin = RSH_HOME / "inst" / "benchmarks" / "harness.R"

        R = R_HOME / "bin" / "R"
        Rscript = R_HOME / "bin" / "Rscript"

        benchmarks = [
            b
            for b in Benchmark.from_folder(bench_dir, extension="R")
            if b.keys.path.parent != bench_dir  # Only files recursed inside
            and (path_filter == "" or path_filter not in str(b.keys.path))
        ]

        conf = (
            suite(
                name="RCPSuite",
                benchmarks=benchmarks,
                parser=RebenchParser(),
                working_directory=CWD,
                command=lambda _, benchmark: (
                    [
                        str(R),
                        "--slave",
                        "--no-restore",
                        "-f",
                        str(harness_bin),
                        "--args",
                    ]
                    + ["--output-dir", str(output)]
                    + ["--runs", str(executions)]
                    + ["--rcp"]
                    + [str(benchmark.keys.path.with_suffix(""))]
                ),
            )
            .time("maximum_resident_size")
            .to_config()
        )
        if runs != 1:
            conf = conf.runs(runs)

        executions = conf.get_executions(params)

        check_namespace(Rscript, "microbenchmark")
        check_namespace(Rscript, "rcp")

        if parallel > 1:
            executor = ParallelExecutor(
                parallel, output / "crash", CsvReporter(output / "result.csv")
            )
        else:
            executor = DefaultExecutor(
                output / "crash", CsvReporter(output / "result.csv")
            )

        with executor:
            executor.execute_all(list(executions))


if __name__ == "__main__":
    rcp_main()
