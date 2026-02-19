import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import sys
import subprocess

import benchr
from benchr import Env
import split_results


def run_cmd(*args, **kwargs):
    return subprocess.run(*args, check=True, **kwargs)


# --------------------------------------

LOCALE: Env = {
    "LC_CTYPE": "en_US.UTF-8",
    "LC_TIME": "en_US.UTF-8",
    "LC_MONETARY": "en_US.UTF-8",
    "LC_PAPER": "en_US.UTF-8",
    "LC_ADDRESS": "C",
    "LC_MEASUREMENT": "en_US.UTF-8",
    "LC_NUMERIC": "C",
    "LC_COLLATE": "en_US.UTF-8",
    "LC_MESSAGES": "en_US.UTF-8",
    "LC_NAME": "C",
    "LC_TELEPHONE": "C",
    "LC_IDENTIFICATION": "C",
}

DEFAULT_ENV: Env = LOCALE | {
    "PIR_OSR": "0",
    "PIR_WARMUP": "10",
    "PIR_GLOBAL_SPECIALIZATION_LEVEL": "0",
    "PIR_DEFAULT_SPECULATION": "0",
    "STATS_USE_RIR_NAMES": "1",
}

# TODO: Full suites
ARE_WE_FAST = [
    ("Mandelbrot", 500),
    ("Bounce", 35),
    ("Storage", 100),
]


SHOOTOUT = [
    (
        "binarytrees",
        "binarytrees",
        9,
    ),
    (
        "fannkuchredux",
        "fannkuch",
        9,
    ),
    (
        "fasta",
        "fasta",
        60000,
    ),
    (
        "fastaredux",
        "fastaredux",
        80000,
    ),
    (
        "knucleotide",
        "knucleotide",
        2000,
    ),
    (
        "mandelbrot_ascii",
        "mandelbrot",
        300,
    ),
    (
        "mandelbrot_naive_ascii",
        "mandelbrot",
        200,
    ),
    (
        "nbody",
        "nbody",
        25000,
    ),
    (
        "nbody_naive",
        "nbody",
        20000,
    ),
    (
        "pidigits",
        "pidigits",
        30,
    ),
    (
        "regexdna",
        "regexdna",
        500000,
    ),
    (
        "reversecomplement",
        "reversecomplement",
        150000,
    ),
    (
        "spectralnorm",
        "spectralnorm",
        1200,
    ),
    (
        "spectralnorm_math",
        "spectralnorm",
        1200,
    ),
]

REAL_THING = [
    ("convolution", 500),
    ("convolution_slow", 1500),
    ("volcano", 1),
    ("flexclust", 5),
]

KAGGLE = [
    "basic-analysis",
    "bolt-driver",
    "london-airbnb",
    "placement",
    "titanic",
]


INPUTS = Path(__file__).resolve() / "inputs"
BENCHMARKS = INPUTS / "Benchmarks"


@dataclass
class Iterations:
    areWeFast: int
    shootout: int
    realThing: int
    kaggle: int
    recommenderlab: int


BENCHMARKS_ITERATIONS = 20
EXAMPLES_ITERATIONS = 15
ITERATIONS = Iterations(
    areWeFast=BENCHMARKS_ITERATIONS,
    shootout=BENCHMARKS_ITERATIONS,
    realThing=BENCHMARKS_ITERATIONS,
    kaggle=EXAMPLES_ITERATIONS,
    recommenderlab=EXAMPLES_ITERATIONS,
)


class Runner:
    Rpath: str
    iterations: Iterations
    default_env: Env

    results_folder: Path
    used_folder: Optional[Path]
    record: bool

    def __init__(
        self,
        Rpath: str,
        iterations: Iterations,
        default_env: Env,
        results_folder: Path,
        used_folder: Optional[Path],
        record: bool,
    ) -> None:
        self.Rpath = Rpath
        self.iterations = iterations
        self.default_env = default_env

        self.results_folder = results_folder
        self.used_folder = used_folder
        self.record = record

    def make_env(self, suite: str, benchmark: str) -> Env:
        env = self.default_env

        if self.record:
            env["STATS_NAME"] = f"{suite}:{benchmark}"

            if env.get("PIR_DEBUG", "").strip() != "":
                compile_log_folder = (
                    self.results_folder / "compile-log" / f"{benchmark}_{suite}_slots"
                )
                compile_log_folder.mkdir(parents=True, exist_ok=True)
                env["PIR_DEBUG_FOLDER"] = str(self.results_folder / benchmark)

        if self.used_folder is not None:
            env |= {"STATS_USED": str(self.used_folder / f"{benchmark}_{suite}_slots")}

        return env

    def areWeFast(self, executor: benchr.Executor):
        for name, arg in ARE_WE_FAST:
            cmd = [
                self.Rpath,
                "harness.r",
                name,
                str(self.iterations.areWeFast),
                str(arg),
            ]
            folder = BENCHMARKS / "areWeFast"
            env = self.make_env("areWeFast", name)

            executor.execute(
                command=cmd,
                folder=folder,
                env=env,
                info={"benchmark": name, "suite": "areWeFast"},
            )

    def shootout(self, executor: benchr.Executor):
        for name, subfolder, arg in SHOOTOUT:
            cmd = [
                self.Rpath,
                "harness.r",
                name,
                str(self.iterations.shootout),
                str(arg),
            ]
            folder = BENCHMARKS / "shootout" / subfolder
            env = self.make_env("shootout", name)

            executor.execute(
                command=cmd,
                folder=folder,
                env=env,
                info={"benchmark": name, "suite": "shootout"},
            )

    def realThing(self, executor: benchr.Executor):
        for name, arg in REAL_THING:
            cmd = [
                self.Rpath,
                "harness.r",
                name,
                str(self.iterations.realThing),
                str(arg),
            ]
            folder = BENCHMARKS / "RealThing"
            env = self.make_env("realThing", name)

            executor.execute(
                command=cmd,
                folder=folder,
                env=env,
                info={"benchmark": name, "suite": "RealThing"},
            )

    def kaggle(self, executor: benchr.Executor):
        for kaggle in KAGGLE:
            if self.record:
                cmd = [self.Rpath, "script.R"]
            else:
                cmd = [
                    self.Rpath,
                    str(INPUTS / "kaggle" / "harness.r"),
                    kaggle,
                    str(self.iterations.kaggle),
                ]
            folder = INPUTS / "kaggle" / kaggle / "code"
            env = self.make_env("kaggle", kaggle)

            executor.execute(
                command=cmd,
                folder=folder,
                env=env,
                info={"benchmark": kaggle, "suite": "kaggle"},
            )

    def recommenderlab(self, executor: benchr.Executor):
        if self.record:
            cmd = [self.Rpath, "runner.r"]
        else:
            cmd = [self.Rpath, "harness.r", str(self.iterations.recommenderlab)]
        folder = INPUTS / "recommenderlab"
        env = self.make_env("recommenderlab", "recommenderlab")

        executor.execute(
            command=cmd,
            folder=folder,
            env=env,
            info={"benchmark": "recommenderlab", "suite": "recommenderlab"},
        )

    def run(self, executor: benchr.Executor):
        self.areWeFast(executor)
        self.shootout(executor)
        self.realThing(executor)
        self.kaggle(executor)
        self.recommenderlab(executor)


def record(
    Rpath: Path,
    results_folder: Path,
    ncores: int,
    env: Env = {},
    explicit_executor: bool = False,
    stats_quiet: bool = True,
    stats_verbose: bool = True,
    pir_debug: str = "",
):
    env = (
        DEFAULT_ENV
        | env
        | {
            "STATS_CSV": str(results_folder / "stats.csv"),
            "STATS_BY_SLOTS": str(results_folder / "stats_by_slots.csv"),
            "STATS_QUIET": "1" if stats_quiet else "0",
            "STATS_VERBOSE": "1" if stats_verbose else "0",
            "PIR_DEBUG": pir_debug,
        }
    )

    compile_log_folder = results_folder / "compile-log"

    if pir_debug != "":
        compile_log_folder.mkdir()

    runner = Runner(
        Rpath=str(Rpath),
        iterations=ITERATIONS,
        default_env=env,
        results_folder=results_folder,
        used_folder=None,
        record=True,
    )

    with benchr.ParallelExecutor(explicit_executor, ncores) as executor:
        runner.run(executor)


def benchmark(
    Rpath: Path,
    experiment_name: str,
    results_folder: Path,
    env: Env = {},
    used_folder: Optional[Path] = None,
    explicit_executor: bool = False,
):
    env = (
        DEFAULT_ENV
        | env
        | {
            "STATS_QUIET": "1",
            "STATS_NO_USED": "1",
        }
    )

    runner = Runner(
        Rpath=str(Rpath),
        iterations=ITERATIONS,
        default_env=env,
        results_folder=results_folder,
        used_folder=used_folder,
        record=False,
    )

    result_csv = (
        f"benchmark_{experiment_name}.tsv" if experiment_name != "" else "benchmark.tsv"
    )

    with benchr.BenchmarkRunner(
        explicit_executor,
        results_folder / result_csv,
        ["benchmark", "suite"],
    ) as executor:
        runner.run(executor)


def rebench(Rpath: Path, results_folder: Path, ncores: int):
    record_folder = results_folder / "record"
    splits_reduced = results_folder / "split-reduced"
    splits_minimal = results_folder / "split-minimal"

    # Record
    record(Rpath, results_folder=results_folder / "record", ncores=ncores)

    # Split
    split_results.split(
        str(record_folder),
        str(splits_reduced),
    )
    # run_cmd(
    #     [
    #         str(Path(__file__).resolve() / ".venv" / "bin" / "python"),
    #         "split_results.py",
    #         str(record_folder),
    #         str(splits_reduced),
    #     ],
    # )

    split_results.split(
        str(record_folder),
        str(splits_minimal),
        "minimal",
    )
    # run_cmd(
    #     [
    #         str(Path(__file__).resolve() / ".venv" / "bin" / "python"),
    #         "split_results.py",
    #         str(record_folder),
    #         str(splits_minimal),
    #         "minimal",
    #     ],
    # )

    # Experiments experiment
    benchmark(Rpath, "baseline", results_folder)
    benchmark(Rpath, "reduced", results_folder, used_folder=splits_reduced)
    benchmark(Rpath, "minimal", results_folder, used_folder=splits_minimal)
    benchmark(Rpath, "no_slots", results_folder, env={"STATS_TF_RECORD": "0"})

    # Interpreter
    interp_env = {"PIR_ENABLE": "off"}

    benchmark(Rpath, "baseline_interp", results_folder)
    benchmark(
        Rpath,
        "reduced_interp",
        results_folder,
        used_folder=splits_reduced,
        env=interp_env,
    )
    benchmark(
        Rpath,
        "minimal_interp",
        results_folder,
        used_folder=splits_minimal,
        env=interp_env,
    )
    benchmark(
        Rpath,
        "no_slots_interp",
        results_folder,
        env=interp_env | {"STATS_TF_RECORD": "0"},
    )


def main():
    p = argparse.ArgumentParser(prog="example")

    p.add_argument(
        "experiment",
        help="The experiment to run",
        choices=["benchmark", "record", "rebench"],
    )

    p.add_argument("Rpath", help="The path to Ř build folder (no bin/R)")
    p.add_argument("results_folder", help="The folder to put results to")
    p.add_argument(
        "--force-results",
        help="If results folder exists, delete it",
        action="store_true",
    )
    p.add_argument(
        "--explicit",
        help="Display explicit information about executed commands",
        action="store_true",
    )
    p.add_argument(
        "--ncores",
        "-j",
        help="Number of processes to run in parallel with (default: 30)",
        type=int,
        default=30,
    )

    penv = p.add_argument_group("Ř reporting")
    penv.add_argument("--stats-quiet", action="store_true")
    penv.add_argument("--stats-verbose", action="store_true")
    penv.add_argument(
        "--pir_debug",
        help="PIR_DEBUG",
        default="PrintEarlyRir,PrintEarlyPir,PrintPirAfterOpt,PrintOptimizationPasses,OmitDeoptBranches,OnlyChanges",
    )

    # Parse
    args = p.parse_args()
    print(args)

    # Common stuff
    Rpath = Path(args.Rpath).resolve()
    results_folder = Path(args.results_folder).resolve()
    explicit_executor = args.explicit

    # Results folder shenanigans
    if results_folder.exists():
        if not args.force_results:
            print(f"ERROR: {results_folder} exists")
            sys.exit(1)

        if results_folder.is_dir():
            shutil.rmtree(results_folder)
        else:
            results_folder.unlink()

    results_folder.mkdir(parents=True, exist_ok=True)

    # Prepare inputs
    run_cmd(str(INPUTS / "prepare_inputs.sh"))

    # Run
    e = args.experiment
    if e == "benchmark":
        benchmark(
            Rpath,
            experiment_name="",
            results_folder=results_folder,
            explicit_executor=explicit_executor,
        )
    elif e == "record":
        record(
            Rpath,
            results_folder=results_folder,
            ncores=args.ncores,
            explicit_executor=explicit_executor,
            stats_quiet=args.stats_quiet,
            stats_verbose=args.stats_verbose,
            pir_debug=args.pir_debug,
        )
    elif e == "rebench":
        rebench(Rpath, results_folder, args.ncores)


if __name__ == "__main__":
    main()
