from pathlib import Path
from typing import Callable, TextIO
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import shutil
import os
import sys

Env = dict[str, str]


class Executor:
    explicit: bool

    def __init__(self, explicit: bool) -> None:
        self.explicit = explicit

    def execute(
        self,
        command: list[str],
        folder: Path,
        env: Env,
        info: dict[str, str],
    ): ...


class ParallelExecutor(Executor):
    pool: ThreadPoolExecutor

    def __init__(self, explicit: bool, ncores: int) -> None:
        super().__init__(explicit)
        self.pool = ThreadPoolExecutor(max_workers=ncores)

    def execute(self, *args, **kwargs):
        self.pool.submit(super().execute, *args, **kwargs)

    def __enter__(self):
        self.pool.__enter__()
        return self

    def __exit__(self, *args):
        return self.pool.__exit__(*args)


class BenchmarkRunner(Executor):
    result: TextIO

    def __init__(self, explicit: bool, result: TextIO) -> None:
        super().__init__(explicit)
        self.result = result

    def execute(
        self,
        command: list[str],
        folder: Path,
        env: Env,
        info: dict[str, str],
    ): ...

    def __enter__(self):
        self.result.__enter__()
        return self

    def __exit__(self, *args):
        return self.result.__exit__(*args)


def benchmark_runner(explicit: bool, result_path: Path) -> BenchmarkRunner:
    # TODO: Check rebench-denoise
    # TODO: write CSV headers
    file = open(result_path, "w")
    try:
        file.write(",".join(csv_headers))
        file.write("\n")
    except:
        file.close()
        raise

    return BenchmarkRunner(explicit=explicit, result=file)


@dataclass
class Benchmark[T]:
    name: str
    data: T


@dataclass
class BenchmarkSuite[T]:
    all_benchmarks: list[Benchmark[T]]
    suite: str

    to_command: Callable[[Benchmark[T]], list[str]]
    folder: Path | Callable[[Benchmark[T]], Path]
    env: Env | Callable[[Benchmark[T]], Env]


Experiment = list[BenchmarkSuite]


def run_benchmark_suite(executor: Executor, suite: BenchmarkSuite, base_env: Env):
    for benchmark in suite.all_benchmarks:
        command = suite.to_command(benchmark)
        folder = suite.folder(benchmark) if callable(suite.folder) else suite.folder
        env = base_env | (suite.env(benchmark) if callable(suite.env) else suite.env)
        info = {
            "benchmark": benchmark.name,
            "suite": suite.suite,
        }

        executor.execute(
            command=command,
            folder=folder,
            env=env,
            info=info,
        )


def run_experiment(executor: Executor, experiment: Experiment, base_env: Env):
    for benchmark_suite in experiment:
        run_benchmark_suite(executor, benchmark_suite, base_env)


# ------------------------------------------------


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


class BenchmarkBuilder:
    Rpath: Path
    record: bool
    results_folder: Path
    compile_log_folder: Path

    def __init__(
        self, Rpath: Path, record: bool, results_folder: Path, compile_log_folder: Path
    ) -> None:
        self.Rpath = Rpath
        self.record = record
        self.results_folder = results_folder
        self.compile_log_folder = compile_log_folder

    def make_benchmark_command_callback(
        self, iterations: int, extra_param: Callable[[Benchmark], str]
    ):
        def make_command(benchmark: Benchmark) -> list[str]:
            return [
                str(self.Rpath),
                "harness.r",
                benchmark.name,
                str(iterations),
                extra_param(benchmark),
            ]

        return make_command

    def make_env_callback(self, suite: str):
        if self.record:

            def record_env(benchmark: Benchmark) -> Env:
                return {
                    "PIR_DEBUG_FOLDER": str(self.results_folder / benchmark.name),
                    "STATS_NAME": f"{suite}:{benchmark.name}",
                    "STATS_USED": str(
                        self.compile_log_folder / f"{benchmark.name}_{suite}_slots"
                    ),
                }

            return record_env
        else:
            return {}

    def areWeFast(self, iterations: int) -> BenchmarkSuite:
        all_benchmarks = list(map(lambda x: Benchmark(x[0], str(x[1])), ARE_WE_FAST))

        return BenchmarkSuite(
            all_benchmarks=all_benchmarks,
            to_command=self.make_benchmark_command_callback(
                iterations, lambda x: str(x.data)
            ),
            suite="areWeFast",
            folder=BENCHMARKS / "areWeFast",
            env=self.make_env_callback("areWeFast"),
        )

    def shootout(self, iterations: int) -> BenchmarkSuite:
        all_benchmarks = list(map(lambda x: Benchmark(x[0], (x[1], x[2])), SHOOTOUT))

        def folder(benchmark: Benchmark) -> Path:
            return BENCHMARKS / "shootout" / benchmark.data[0]

        return BenchmarkSuite(
            all_benchmarks=all_benchmarks,
            to_command=self.make_benchmark_command_callback(
                iterations, lambda x: str(x.data[1])
            ),
            suite="shootout",
            folder=folder,
            env=self.make_env_callback("shootout"),
        )

    def realThing(self, iterations: int) -> BenchmarkSuite:
        all_benchmarks = list(map(lambda x: Benchmark(x[0], x[1]), REAL_THING))

        return BenchmarkSuite(
            all_benchmarks=all_benchmarks,
            to_command=self.make_benchmark_command_callback(
                iterations, lambda x: str(x.data)
            ),
            suite="realThing",
            folder=BENCHMARKS / "RealThing",
            env=self.make_env_callback("realThing"),
        )

    def kaggles(self, iterations: int) -> BenchmarkSuite:
        all_benchmarks = list(map(lambda x: Benchmark(x, None), KAGGLE))

        def to_command(benchmark: Benchmark) -> list[str]:
            if self.record:
                return [str(self.Rpath), "script.R"]
            else:
                return [
                    str(self.Rpath),
                    str(INPUTS / "kaggle" / "harness.r"),
                    benchmark.name,
                    str(iterations),
                ]

        def folder(benchmark: Benchmark) -> Path:
            return INPUTS / "kaggle" / benchmark.name / "code"

        return BenchmarkSuite(
            all_benchmarks=all_benchmarks,
            to_command=to_command,
            suite="kaggle",
            folder=folder,
            env=self.make_env_callback("kaggle"),
        )

    def recommenderlab(self, iterations: int) -> BenchmarkSuite:
        def to_command(_: Benchmark) -> list[str]:
            if self.record:
                return [str(self.Rpath), "runner.r"]
            else:
                return [str(self.Rpath), "harness.r", str(iterations)]

        return BenchmarkSuite(
            all_benchmarks=[Benchmark("recommenderlab", None)],
            to_command=to_command,
            suite="recommenderlab",
            folder=INPUTS / "recommenderlab",
            env=self.make_env_callback("recommenderlab"),
        )


def build_experiment(
    Rpath: Path,
    benchmark_iterations: int,
    example_iterations: int,
    record: bool,
    results_folder: Path,
    compile_log_folder: Path,
) -> list[BenchmarkSuite]:
    builder = BenchmarkBuilder(Rpath, record, results_folder, compile_log_folder)

    return [
        builder.areWeFast(benchmark_iterations),
        builder.shootout(benchmark_iterations),
        builder.realThing(benchmark_iterations),
        builder.kaggles(example_iterations),
        builder.recommenderlab(example_iterations),
    ]


BENCHMARKS_ITERATIONS = 20
EXAMPLES_ITERATIONS = 15


def record(
    Rpath: Path,
    results_folder: Path,
    explicit_executor: bool,
    ncores: int,
    explicit_stats: bool,
    verbose_stats: bool,
    pir_debug: str,
):
    # TODO: in main
    if results_folder.exists():
        if results_folder.is_dir():
            shutil.rmtree(results_folder)
        else:
            results_folder.unlink()

    results_folder.mkdir(parents=True, exist_ok=True)

    # PIR_DEBUG
    # "PrintEarlyRir,PrintEarlyPir,PrintPirAfterOpt,PrintOptimizationPasses,OmitDeoptBranches,OnlyChanges",
    # STAT_QUIET 0
    # STATS_VERBOSE 0

    env = DEFAULT_ENV | {
        "STATS_CSV": str(results_folder / "stats.csv"),
        "STATS_BY_SLOTS": str(results_folder / "stats_by_slots.csv"),
        "STATS_QUIET": "1" if explicit_stats else "0",
        "STATS_VERBOSE": "1" if verbose_stats else "0",
        "PIR_DEBUG": pir_debug,
    }

    compile_log_folder = results_folder / "compile-log"

    if pir_debug != "":
        compile_log_folder.mkdir()

    experiment = build_experiment(
        Rpath,
        benchmark_iterations=BENCHMARKS_ITERATIONS,
        example_iterations=EXAMPLES_ITERATIONS,
        record=True,
        results_folder=results_folder,
        compile_log_folder=compile_log_folder,
    )

    with ParallelExecutor(explicit_executor, ncores) as executor:
        run_experiment(executor, experiment, env)


BENCHMARK_ENV = {
    "STATS_QUIET": "1",
    "STATS_NO_USED": "1",
}


def benchmark(
    Rpath: Path,
    experiment_name: str,
    results_folder: Path,
    explicit_executor: bool,
):
    env = DEFAULT_ENV | BENCHMARK_ENV

    experiment = build_experiment(
        Rpath,
        benchmark_iterations=BENCHMARKS_ITERATIONS,
        example_iterations=EXAMPLES_ITERATIONS,
        record=False,
        results_folder=results_folder,
        compile_log_folder=None,
    )

    with benchmark_runner(
        explicit_executor, results_folder / f"benchmark_{experiment_name}.tsv"
    ) as executor:
        run_experiment(executor, experiment, env)


def main(): ...


if __name__ == "__main__":
    main()
