import abc
import argparse
import dataclasses
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from pprint import pprint
from threading import Lock
from types import SimpleNamespace
from typing import Any, Callable, Optional

# --------------------------------------
#           DEFINITIONS
# --------------------------------------

Env = dict[str, str]
Command = list[str]


class Parameters(SimpleNamespace):
    def __or__(self, other: "Parameters", /) -> "Parameters":
        return Parameters(**vars(self), **vars(other))


# --------------------------------------
#           HELPER
# --------------------------------------


# TODO: maybe adjust api, maybe remove
def run_cmd(*args, **kwargs):
    """
    Run a very simple command, wrapper around subprocess.run
    """
    return subprocess.run(*args, check=True, **kwargs)


# --------------------------------------
#          INPUT DEFINITIONS
# --------------------------------------


@dataclass
class Run:
    benchmark_name: str
    suite: str

    command: Command
    working_directory: Path
    env: Env
    info: dict[str, str]

    def as_identifier(self) -> str:
        id = f"{self.suite}, {self.benchmark_name}"

        some_info = False
        for k, v in self.info.items():
            if not some_info:
                id += " ("
                some_info = True
            else:
                id += ", "
            id += f"{k}={v}"

        if some_info:
            id += ")"

        return id


@dataclass
class Benchmark:
    """
    A definition of one benchmark. data can be any benchmark-specific data
    that are needed for its execution.
    """

    name: str
    data: tuple[Any, ...] | Any

    def __init__(self, name: str, *data: Any) -> None:
        self.name = name

        if len(data) == 1:
            self.data = data[0]
        else:
            self.data = data


B = Benchmark

type SuiteFactory[T] = Callable[[Parameters, Benchmark], T]


# TODO: Maybe allow benchmark to be only string
@dataclass
class Suite:
    """
    A collection of benchmarks. They should all be connected with similar
    structure.
    """

    name: str
    benchmarks: list[Benchmark]

    command: SuiteFactory[Command]

    working_directory: Optional[SuiteFactory[Path] | Path] = None
    env: SuiteFactory[Env] | Env = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if len(self.benchmarks) == 0:
            raise ValueError(f"No benchmarks defined in {self.name}!")

    def mk_command(self, parameters: Parameters, benchmark: Benchmark) -> Command:
        return self.command(parameters, benchmark)


type ConfigFactory[T] = Callable[[Parameters, Suite, Benchmark], T]


def default_info(p: Parameters, s: Suite, b: Benchmark) -> dict[str, str]:
    return {}


@dataclass
class Config:
    """
    The full configuration of all suites.
    """

    suites: list[Suite]

    working_directory: Optional[ConfigFactory[Path] | Path]
    env: ConfigFactory[Env] | Env

    info: ConfigFactory[dict[str, str]]

    def __init__(
        self,
        *suites: Suite,
        working_directory: Optional[ConfigFactory[Path] | Path] = None,
        env: ConfigFactory[Env] | Env = {},
        info: ConfigFactory[dict[str, str]] = default_info,
    ) -> None:
        if len(suites) == 0:
            raise ValueError("No suites defined!")

        self.suites = list(suites)
        self.working_directory = working_directory
        self.env = env
        self.info = info


def config_to_runs(config: Config, parameters: Parameters) -> list[Run]:
    results = []

    for suite in config.suites:
        for bench in suite.benchmarks:
            command = suite.mk_command(parameters, bench)

            # WD
            if suite.working_directory is not None:
                if callable(suite.working_directory):
                    wd = suite.working_directory(parameters, bench)
                else:
                    wd = suite.working_directory

            elif config.working_directory is not None:
                if callable(config.working_directory):
                    wd = config.working_directory(parameters, suite, bench)
                else:
                    wd = config.working_directory
            else:
                raise ValueError(f"No working directory defined for suite {suite.name}")

            # Env
            if callable(suite.env):
                env = suite.env(parameters, bench)
            else:
                env = suite.env

            if callable(config.env):
                env |= config.env(parameters, suite, bench)
            else:
                env |= config.env

            info = config.info(parameters, suite, bench)

            results.append(
                Run(
                    benchmark_name=bench.name,
                    suite=suite.name,
                    command=command,
                    working_directory=wd,
                    env=env,
                    info=info,
                )
            )

    return results


# --------------------------------------
#           RESULT DEFINITIONS
# --------------------------------------


@dataclass
class Measurement:
    value: float
    unit: str


@dataclass
class Iteration:
    runtime: Optional[float] = None
    criterions: dict[str, list[Measurement]] = dataclasses.field(default_factory=dict)

    def add_criterion(self, criterion: str, measurement: Measurement):
        if criterion not in self.criterions:
            self.criterions[criterion] = [measurement]
        else:
            self.criterions[criterion].append(measurement)

    def __or__(self, other: "Iteration") -> "Iteration":
        result = Iteration(
            runtime=self.runtime if self.runtime is not None else other.runtime,
            criterions={k: v for k, v in self.criterions.items()},
        )

        for k, v in other.criterions.items():
            if k in result.criterions:
                result.criterions[k].extend(v)
            else:
                result.criterions[k] = v

        return result


@dataclass
class RunResult:
    run: Run
    iterations: list[Iteration] = dataclasses.field(default_factory=list)

    def new_iteration(self, *args, **kwargs) -> Iteration:
        it = Iteration(*args, **kwargs)
        self.iterations.append(it)
        return it

    def __or__(self, other: "RunResult") -> "RunResult":
        assert self.run == other.run

        return RunResult(
            run=self.run,
            iterations=[
                x | y
                for x, y in zip_longest(
                    self.iterations, other.iterations, fillvalue=Iteration()
                )
            ],
        )


# --------------------------------------
#           PARSER
# --------------------------------------


class ResultParser(abc.ABC):
    @abc.abstractmethod
    def parse(self, run: Run, stdout: str) -> RunResult: ...


class PlainSecondsParser(ResultParser):
    def parse(self, run: Run, stdout: str) -> RunResult:
        result = RunResult(run)

        for line in stdout.split("\n"):
            try:
                time = float(line) * 1000
                result.new_iteration(runtime=time)
            except ValueError:
                pass

        return result


class LastLineParser(ResultParser):
    subparser: ResultParser

    def __init__(self, subparser: ResultParser) -> None:
        self.subparser = subparser

    def parse(self, run: Run, stdout: str) -> RunResult:
        for line in reversed(stdout.split("\n")):
            if line.strip() != "":
                return self.subparser.parse(run, line)

        return RunResult(run)


RUNTIME_CRITERION = "runtime"


class RegexParser(ResultParser):
    criterion: str
    regex: re.Pattern[str]
    match_group: str | int

    def __init__(self, criterion: str, regex: str, match_group: str | int) -> None:
        self.criterion = criterion
        self.regex = re.compile(regex)
        self.match_group = match_group

    def parse(self, run: Run, stdout: str) -> RunResult:
        result = RunResult(run)

        for line in stdout.split("\n"):
            match = self.regex.match(line)
            if match is not None:
                value = float(match.group(self.match_group))

                if self.criterion == RUNTIME_CRITERION:
                    result.new_iteration(runtime=value)
                else:
                    it = result.new_iteration()
                    it.add_criterion(self.criterion, Measurement(value, ""))

        return result


class RebenchParser(ResultParser):
    """
    Format used by the ReBench (https://github.com/smarr/ReBench) benchmarker,
    mostly copied from the RebenchLogAdapter. The supported format is:
    ```
    optional_prefix: benchmark_name optional_criterion: iterations=123 runtime: 1000[ms|us]
    ```
    or for non-runtime
    ```
    optional_prefix: benchmark_name: criterion: number_with_unit
    ```

    Unlike ReBench, benchr only reports runtime in ms. Runtime report with other
    criterion other than "total" (or none) are ignored.

    When a runtime with no criterion (or criterion "total") or non-runtime
    criterion "total" is parsed, a new iteration is assumed. This should be
    equivalent to ReBench.
    """

    re_log_line = re.compile(
        r"^(?:.*: )?([^\s]+)( [\w\.]+)?: iterations=([0-9]+) "
        + r"runtime: (?P<runtime>(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)"
        + r"(?P<unit>[mu])s"
    )

    re_extra_criterion_log_line = re.compile(
        r"^(?:.*: )?([^\s]+): (?P<criterion>[^:]{1,30}):\s*"
        + r"(?P<value>(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)"
        + r"(?P<unit>[a-zA-Z]+)"
    )

    def parse(self, run: Run, stdout: str) -> RunResult:
        result = RunResult(run)
        current_iteration = None

        for line in stdout.split("\n"):
            match = self.re_log_line.match(line)
            if match is not None:
                # Match runtime
                time = float(match.group("runtime"))
                if match.group("unit") == "u":
                    time /= 1000

                # Match criterion, maybe skip
                criterion = match.group(2)
                if criterion is not None and criterion.strip() != "total":
                    continue

                # Add time, force new iteration
                if current_iteration is None:
                    result.new_iteration(runtime=time)
                else:
                    current_iteration.runtime = time
                    current_iteration = None

                continue

            match = self.re_extra_criterion_log_line.match(line)
            if match is not None:
                # Match groups
                value = float(match.group("value"))
                unit = match.group("unit")
                criterion = match.group("criterion")

                if current_iteration is None:
                    current_iteration = result.new_iteration()

                # Add measurement
                measure = Measurement(value, unit)
                current_iteration.add_criterion(criterion, measure)

                # Force new iteration
                if measure == "total":
                    current_iteration = None
                continue

        return result


class MixedResultParser(ResultParser):
    parsers: list[ResultParser]

    def __init__(self, *parsers: ResultParser) -> None:
        self.parsers = list(*parsers)

    def parse(self, run: Run, stdout: str) -> RunResult:
        result = RunResult(run)

        for parser in self.parsers:
            result |= parser.parse(run, stdout)

        return result


# --------------------------------------
#           TUI
# --------------------------------------


class TUI:
    if sys.stdout.isatty():
        # TODO:
        # RESET_LINE = "\r"
        RESET_LINE = "\n"
        RESET = "\033[0m"
        BOLD = "\033[1m"
        BLACK = "\033[30m"
        RED = "\033[31m"
        GREEN = "\033[32m"
        YELLOW = "\033[33m"
        BLUE = "\033[34m"
        MAGENTA = "\033[35m"
        CYAN = "\033[36m"
        WHITE = "\033[37m"
    else:
        RESET_LINE = "\n"
        RESET = ""
        BOLD = ""
        BLACK = ""
        RED = ""
        GREEN = ""
        YELLOW = ""
        BLUE = ""
        MAGENTA = ""
        CYAN = ""
        WHITE = ""


# --------------------------------------
#           FORMATTERS
# --------------------------------------


class Formatter(abc.ABC):
    @abc.abstractmethod
    def format(self, results: list[RunResult]): ...

    @staticmethod
    def get_all_extra_criterions(results: list[RunResult]) -> list[str]:
        out = []

        for result in results:
            for iter in result.iterations:
                for crit in iter.criterions.keys():
                    if crit not in out:
                        out.append(crit)

        return out

    @staticmethod
    def get_all_info_columns(results: list[RunResult]) -> list[str]:
        out = []

        for result in results:
            for i in result.run.info.keys():
                if i not in out:
                    out.append(i)

        return out


class CsvFormatter(Formatter):
    filepath: Path
    separator: str

    def __init__(self, filepath: Path, separator: str = ",") -> None:
        self.filepath = filepath
        self.separator = separator

    @staticmethod
    def escape_text(text: str) -> str:
        if "," in text:
            return '"' + text.replace('"', r"\"") + '"'

        return text

    def format_line(self, line: list[str]) -> str:
        return self.separator.join(map(self.escape_text, line)) + "\n"

    def format(self, results: list[RunResult]):
        info_cols = Formatter.get_all_info_columns(results)
        extra_criterions = Formatter.get_all_extra_criterions(results)

        columns = (
            ["benchmark", "suite"]
            + info_cols
            + ["iteration", "runtime"]
            + extra_criterions
        )

        with open(self.filepath, "wt") as file:
            file.write(self.format_line(columns))

            for result in results:
                iter_counter = 0
                for iter in result.iterations:
                    line: list[str] = [result.run.benchmark_name, result.run.suite]

                    for ic in info_cols:
                        line.append(result.run.info.get(ic, ""))

                    line += [
                        str(iter_counter),
                        str(iter.runtime) if iter.runtime else "",
                    ]

                    for ec in extra_criterions:
                        if ec in iter.criterions:
                            criter = iter.criterions[ec]
                            # TODO: handle multiple datapoints
                            line.append(str(criter[0].value))
                        else:
                            line.append("")

                    file.write(self.format_line(line))
                    iter_counter += 1


# --------------------------------------
#           EXECUTORS
# --------------------------------------


class Executor(abc.ABC):
    @abc.abstractmethod
    def execute(self, run: Run): ...

    def execute_all(self, runs: list[Run]):
        for run in runs:
            self.execute(run)


class DefaultExecutor(Executor):
    """
    The main Executor, which executes given commands, reporting success or
    failures to reporter
    """

    finished_runs: int
    failed_runs: int

    parser: ResultParser
    formatter: Formatter

    results: list[RunResult]

    def __init__(self, parser: ResultParser, formatter: Formatter) -> None:
        self.finished_runs = 0
        self.failed_runs = 0

        self.parser = parser
        self.formatter = formatter
        self.results = []

    def execute(self, run: Run):
        cmd = shutil.which(run.command[0])
        if cmd is None:
            self.error_run(run, f"Command not found ({run.command[0]})")
            return

        run.command[0] = cmd

        self.start_run(run)

        proc = None
        try:
            # TODO: group, process group?
            proc = subprocess.run(
                run.command,
                # run spec
                capture_output=True,
                check=False,
                # Popen spec
                stdin=None,
                shell=False,
                cwd=run.working_directory,
                env=run.env,
                text=True,
            )

            if proc.returncode != 0:
                self.error_run(
                    run, f"Program ended with non-zero return code ({proc.returncode})"
                )
                return

            self.finalize(
                run,
                stdout=proc.stdout,
                stderr=proc.stderr,
            )

        except OSError as e:
            self.error_run(run, str(e))

    def start_run(self, run: Run) -> None:
        print(
            f"{TUI.RESET_LINE}["
            + f"{TUI.RED}{TUI.BOLD}{self.failed_runs}{TUI.RESET}/"
            + f"{TUI.GREEN}{TUI.BOLD}{self.finished_runs}{TUI.RESET}] "
            + run.as_identifier(),
            end="",
        )

    def error_run(self, run: Run, msg: str):
        self.failed_runs += 1
        print(
            f"\n{TUI.RED}{TUI.BOLD}Error in {run.as_identifier()}{TUI.RESET}\n{msg}\n"
        )

    def finalize(self, run: Run, stdout: str, stderr: str) -> None:
        self.finished_runs += 1
        self.results.append(self.parser.parse(run, stdout))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.formatter.format(self.results)
        return False


#
# class ParallelExecutor(DefaultExecutor):
#     pool: ThreadPoolExecutor
#     lock: Lock
#
#     in_process_runs: int
#
#     def __init__(self, ncores: int) -> None:
#         super().__init__()
#         self.pool = ThreadPoolExecutor(max_workers=ncores)
#         self.lock = Lock()
#
#     def execute(self, run: Run):
#         self.pool.submit(super().execute, run)
#
#     def __enter__(self):
#         return self
#
#     def __exit__(self, *args):
#         self.pool.shutdown(wait=True)
#         super().__exit__(*args)
#         return False


class DryExecutor(Executor):
    """
    Simple executor which only prints what would be executed
    """

    def execute(self, run: Run):
        print("Run:", " ".join(run.command))
        print("Working working_directory:", str(run.working_directory))
        print("Environment: ", end="")
        pprint(run.env)
        print("Info: ", end="")
        pprint(run.info)
        print("-" * 10)


# --------------------------------------
#          ARGUMENT PARSING
# --------------------------------------


def make_argparser(*params: str, **kwarg_params: Any) -> argparse.ArgumentParser:
    """
    Create a default argument parser from the given parameters
    """
    parser = argparse.ArgumentParser()
    user = parser.add_argument_group("User specified parameters")

    for p in params:
        user.add_argument(
            f"--{p}",
            metavar="str",
            type=str,
            required=True,
            dest=p,
        )

    for k, v in kwarg_params.items():
        t = type(v) if v is not None else str
        user.add_argument(
            f"--{k}",
            help=f"(Default: {v})",
            metavar=t.__name__,
            type=t,
            default=v,
            dest=k,
        )

    return parser


def namespace_to_parameters(ns: argparse.Namespace) -> Parameters:
    """
    Convert the argparse.Namespace to Parameters
    """
    return Parameters(**vars(ns))


def parse_params(*params: str, **kwarg_params: Any) -> Parameters:
    """
    Create a default argument parser and run it on argv
    """
    parser = make_argparser(*params, **kwarg_params)
    args = parser.parse_args()
    return namespace_to_parameters(args)


# --------------------------------------
#          DEFAULT RUN
# --------------------------------------


def main(config: Config, *params: str, **kwarg_params: Any) -> None:
    """
    Sane default main. config is the benchmarks configuration that will be
    executed, params is a list of required parameters from the user,
    kwarg_params are optional parameters with their default value.
    """
    parser = make_argparser(*params, **kwarg_params)

    defp = parser.add_argument_group("Default benchr parameters")
    defp.add_argument(
        "--output",
        help="Where to store the results (Default: ./output)",
        metavar="file",
        type=str,
        default="./output",
        dest="__output",
    )
    defp.add_argument(
        "--jobs",
        "-j",
        help="Allow this many runs in parallel (Default: 1)",
        metavar="jobs",
        type=int,
        default=1,
        dest="__jobs",
    )
    defp.add_argument(
        "--dry",
        help="Do not run, only print what would be runned",
        action="store_true",
        dest="__dry",
    )

    ps = namespace_to_parameters(parser.parse_args())

    runs = config_to_runs(config, ps)

    if ps.__dry:
        DryExecutor().execute_all(runs)
    elif ps.__jobs > 1:
        # TODO: parallel executor
        pass
    else:
        with DefaultExecutor(
            LastLineParser(PlainSecondsParser()),
            CsvFormatter(Path(ps.__output).resolve()),
        ) as executor:
            executor.execute_all(runs)


# TODO: Catch keyboard_interrupts
