import abc
import argparse
import dataclasses
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from threading import Lock
from types import SimpleNamespace
from typing import Any, Callable, Iterator, Literal, Optional, Sequence

# --------------------------------------
#           HELPERS
# --------------------------------------


def const[T](x: T) -> Callable[..., T]:
    return lambda *args, **kwargs: x


# --------------------------------------
#           DEFINITIONS
# --------------------------------------

Env = dict[str, str]
Command = list[str]


class Parameters(SimpleNamespace):
    def __or__(self, other: "Parameters") -> "Parameters":
        return Parameters(**vars(self), **vars(other))

    @staticmethod
    def from_namespace(ns: argparse.Namespace) -> "Parameters":
        """
        Convert the argparse.Namespace to Parameters
        """
        return Parameters(**vars(ns))


TimeBinutilColumns = Literal[
    "maximum_resident_size",
    "average_resident_size",
    "user_time",
    "system_time",
    "clock_time",
]


# --------------------------------------
#          INPUT DEFINITIONS
# --------------------------------------


@dataclass
class Execution:
    """
    A definition of ready-to-run command
    """

    benchmark_name: str
    suite: str
    parser: "ResultParser"

    command: Command
    working_directory: Path
    env: Env

    info: dict[str, str]

    def as_identifier(self) -> str:
        id = f"{self.suite},{self.benchmark_name}"

        if len(self.info) != 0:
            id += " ("
            id += ", ".join((f"{k}={v}" for k, v in self.info.items()))
            id += ")"

        return id

    @dataclass
    class Incomplete:
        benchmark_name: str
        suite: str
        parser: Optional["ResultParser"]

        command: Optional[Command]
        working_directory: Optional[Path]
        env: Env

        info: dict[str, str]

        def finalize(self) -> "Execution":
            if self.parser is None:
                raise ValueError(
                    f"Benchmark {self.benchmark_name} in suite {self.suite} is missing a parser"
                )

            if self.working_directory is None:
                raise ValueError(
                    f"Benchmark {self.benchmark_name} in suite {self.suite} is missing working directory"
                )

            if self.command is None:
                raise ValueError(
                    f"Benchmark {self.benchmark_name} in suite {self.suite} is missing a command"
                )

            return Execution(
                benchmark_name=self.benchmark_name,
                suite=self.suite,
                parser=self.parser,
                command=self.command,
                working_directory=self.working_directory,
                env=self.env,
                info=self.info,
            )


@dataclass
class Benchmark:
    """
    A definition of one benchmark. data and keys can be any benchmark-specific
    data that are needed for its execution.
    """

    name: str
    data: tuple[Any, ...] | Any
    keys: SimpleNamespace

    def __init__(self, name: str, *data: Any, **keys: Any) -> None:
        self.name = name

        if len(data) == 1:
            self.data = data[0]
        else:
            self.data = data

        self.keys = SimpleNamespace(keys)

    @staticmethod
    def from_files(*files: Path) -> list["Benchmark"]:
        return [
            Benchmark(
                file.stem,
                path=file,
            )
            for file in files
        ]

    @staticmethod
    def from_folder(folder: Path, extension: Optional[str] = None) -> list["Benchmark"]:
        res = []
        for path, _, files in folder.walk():
            for file in files:
                p = path / file
                if extension is None or p.suffix.lower() == ("." + extension.lower()):
                    res.append(path / file)

        return Benchmark.from_files(*res)


B = Benchmark

# --------------------------------------
#          SUITES OF BENCHMARKS
# --------------------------------------


class BenchmarkCollection[This](abc.ABC):
    @abc.abstractmethod
    def apply_suite_decorator(
        self, decorator: Callable[["Suite"], "Suite"]
    ) -> This: ...

    def runs(self, value: int) -> This:
        return self.matrix("run", *range(1, value + 1))

    def matrix[T](
        self,
        name: str,
        *parameters: T,
        working_directory: Optional[Callable[[T], Path]] = None,
        env: Optional[Callable[[T], Env]] = None,
    ) -> This:
        return self.apply_suite_decorator(
            lambda suite: MatrixSuite(
                name,
                suite,
                parameters,
                working_directory,
                env,
            )
        )

    def time(
        self, *columns: TimeBinutilColumns, time_bin: Optional[str] = None
    ) -> This:
        return self.apply_suite_decorator(
            lambda suite: TimeSuite(
                suite,
                list(columns),
                time_bin=time_bin,
            )
        )


class Suite(BenchmarkCollection["Suite"]):
    """
    A collection of benchmarks. They should all be connected with similar
    structure.
    """

    def apply_suite_decorator(self, decorator: Callable[["Suite"], "Suite"]) -> "Suite":
        return decorator(self)

    @abc.abstractmethod
    def get_executions(
        self, parameters: Parameters
    ) -> Iterator[Execution.Incomplete]: ...

    def to_config(self) -> "Config":
        return Config([self])


class BaseSuite(Suite):
    name: str
    benchmarks: Callable[[Parameters], list[Benchmark]]

    command: Optional[Callable[[Parameters, Benchmark], Command]]
    working_directory: Optional[Callable[[Parameters, Benchmark], Path]]
    env: Callable[[Parameters, Benchmark], Env]

    parser: Optional["ResultParser"]

    def __init__(
        self,
        name: str,
        benchmarks: Callable[[Parameters], list[Benchmark]],
        command: Optional[Callable[[Parameters, Benchmark], Command]],
        working_directory: Optional[Callable[[Parameters, Benchmark], Path]],
        env: Callable[[Parameters, Benchmark], Env],
        parser: Optional["ResultParser"],
    ) -> None:
        self.name = name
        self.benchmarks = benchmarks

        self.command = command
        self.working_directory = working_directory
        self.env = env

        self.parser = parser

    def get_executions(self, parameters: Parameters) -> Iterator[Execution.Incomplete]:
        benchs = self.benchmarks(parameters)

        if len(benchs) == 0:
            raise ValueError(f"No benchmarks defined in {self.name}!")

        for b in benchs:
            if self.command is not None:
                command = self.command(parameters, b)
            else:
                command = None

            if self.working_directory is not None:
                working_directory = self.working_directory(parameters, b)
            else:
                working_directory = None

            env = self.env(parameters, b)

            yield Execution.Incomplete(
                benchmark_name=b.name,
                suite=self.name,
                command=command,
                parser=self.parser,
                working_directory=working_directory,
                env=env,
                info={},
            )


def suite(
    name: str,
    benchmarks: Sequence[Benchmark | str] | Callable[[Parameters], list[Benchmark]],
    *,
    command: Optional[Callable[[Parameters, Benchmark], Command]],
    working_directory: Optional[Callable[[Parameters, Benchmark], Path] | Path] = None,
    env: Callable[[Parameters, Benchmark], Env] | Env = {},
    parser: Optional["ResultParser"] = None,
) -> Suite:
    """
    Flexible way of constructing a Suite
    """
    if not callable(benchmarks):
        benchmarks = const(
            [Benchmark(b) if isinstance(b, str) else b for b in benchmarks]
        )

    if working_directory is not None and not callable(working_directory):
        working_directory = const(working_directory)

    if not callable(env):
        env = const(env)

    return BaseSuite(
        name=name,
        benchmarks=benchmarks,
        command=command,
        working_directory=working_directory,
        env=env,
        parser=parser,
    )


# --------------------------------------
#          SUITE DECORATORS
# --------------------------------------


class SuiteDecorator(Suite):
    parent: Suite

    def __init__(self, parent: Suite) -> None:
        self.parent = parent

    def get_executions(self, parameters: Parameters) -> Iterator[Execution.Incomplete]:
        for pexe in self.parent.get_executions(parameters):
            for exe in self.extend_execution(parameters, pexe):
                yield exe

    @abc.abstractmethod
    def extend_execution(
        self, parameters: Parameters, execution: Execution.Incomplete
    ) -> Iterator[Execution.Incomplete]: ...


class MatrixSuite[T](SuiteDecorator):
    name: str
    parameters: Sequence[T]

    matrix_working_directory: Optional[Callable[[T], Path]]
    matrix_env: Callable[[T], Env]

    def __init__(
        self,
        name: str,
        parent: Suite,
        parameters: Sequence[T],
        working_directory: Optional[Callable[[T], Path]] = None,
        env: Optional[Callable[[T], Env]] = None,
    ) -> None:
        super().__init__(parent)

        self.name = name
        self.parameters = parameters

        self.matrix_working_directory = working_directory

        if env is None:
            env = const({})
        self.matrix_env = env

    def extend_execution(
        self, parameters: Parameters, execution: Execution.Incomplete
    ) -> Iterator[Execution.Incomplete]:
        for p in self.parameters:
            e = execution.env | self.matrix_env(p)

            if self.matrix_working_directory is not None:
                wd = self.matrix_working_directory(p)
            else:
                wd = execution.working_directory

            i = execution.info | {self.name: str(p)}

            yield dataclasses.replace(
                execution,
                env=e,
                working_directory=wd,
                info=i,
            )


class TimeSuite(SuiteDecorator):
    parent: Suite
    parser: "ResultParser"
    format: str
    time_bin: str

    def __init__(
        self,
        parent: Suite,
        columns: list[TimeBinutilColumns],
        time_bin: Optional[str] = None,
    ) -> None:
        super().__init__(parent)

        self.parser = time_parser(columns)
        self.format = TimeSuite.make_format(columns)

        if time_bin is None:
            time_bin = shutil.which("time")
            if time_bin is None:
                raise ValueError("Unable to find `time` binary")

        self.time_bin = time_bin

    @staticmethod
    def make_format(columns: list[TimeBinutilColumns]) -> str:
        format = ""
        for col in columns:
            if col == "maximum_resident_size":
                format += "maximum_resident_size: %M\n"

            elif col == "average_resident_size":
                format += "average_resident_size: %t\n"

            elif col == "user_time":
                format += "user_time: %U\n"

            elif col == "system_time":
                format += "system_time: %S\n"

            elif col == "clock_time":
                format += "clock_time: %e\n"

            else:
                raise ValueError(f"Unknown Time column {col}")

        return format

    def extend_execution(
        self, parameters: Parameters, execution: Execution.Incomplete
    ) -> Iterator[Execution.Incomplete]:
        if execution.command is None:
            raise ValueError("TimeSuite cannot extend an empty command")

        execution.command = [self.time_bin, "-f", self.format] + execution.command
        if execution.parser is None:
            execution.parser = self.parser
        else:
            execution.parser = MixedResultParser(execution.parser, self.parser)
        yield execution


# --------------------------------------
#          CONFIGURATION
# --------------------------------------


@dataclass
class Config(BenchmarkCollection["Config"]):
    """
    The full configuration of all suites.
    """

    suites: list[Suite]

    default_parser: Optional["ResultParser"] = None
    default_command: Optional[Callable[[Parameters, Execution.Incomplete], Command]] = (
        None
    )
    default_working_directory: Optional[
        Callable[[Parameters, Execution.Incomplete], Path]
    ] = None
    default_env: Callable[[Parameters, Execution.Incomplete], Env] = const({})

    def command(
        self,
        default_command: Optional[
            Callable[[Parameters, Execution.Incomplete], Command]
        ],
    ) -> "Config":
        return dataclasses.replace(self, default_command=default_command)

    def working_directory(
        self,
        default_working_directory: Callable[[Parameters, Execution.Incomplete], Path],
    ) -> "Config":
        return dataclasses.replace(
            self, default_working_directory=default_working_directory
        )

    def env(
        self,
        default_env: Callable[[Parameters, Execution.Incomplete], Env],
    ) -> "Config":
        return dataclasses.replace(self, default_env=default_env)

    def get_executions(self, parameters: Parameters) -> list[Execution]:
        res = []
        for suite in self.suites:
            for exe in suite.get_executions(parameters):
                if exe.parser is None:
                    if self.default_parser is None:
                        raise ValueError(
                            f"No result parser for benchmark {exe.benchmark_name} in suite {exe.suite}"
                        )

                    exe.parser = self.default_parser

                if exe.command is None:
                    if self.default_command is None:
                        raise ValueError(
                            f"No command for benchmark {exe.benchmark_name} in suite {exe.suite}"
                        )

                    exe.command = self.default_command(parameters, exe)

                if exe.working_directory is None:
                    if self.default_working_directory is None:
                        raise ValueError(
                            f"No working directory for benchmark {exe.benchmark_name} in suite {exe.suite}"
                        )

                    exe.working_directory = self.default_working_directory(
                        parameters, exe
                    )

                exe.env = self.default_env(parameters, exe) | exe.env

                res.append(
                    Execution(
                        benchmark_name=exe.benchmark_name,
                        suite=exe.suite,
                        parser=exe.parser,
                        command=exe.command,
                        working_directory=exe.working_directory,
                        env=exe.env,
                        info=exe.info,
                    )
                )

        return res

    def apply_suite_decorator(
        self, decorator: Callable[["Suite"], "Suite"]
    ) -> "Config":
        return dataclasses.replace(self, suites=list(map(decorator, self.suites)))


# --------------------------------------
#           RESULT DEFINITIONS
# --------------------------------------


@dataclass
class Measurement:
    execution: Execution
    metric: str
    value: float
    unit: str

    measurement_info: dict[str, str]

    @staticmethod
    def runtime(
        execution: Execution,
        value: float,
        unit: str,
        measurement_info: dict[str, str] = {},
    ) -> "Measurement":
        return Measurement(
            execution=execution,
            metric="runtime",
            value=value,
            unit=unit,
            measurement_info=measurement_info,
        )


@dataclass
class ExecutionResult:
    measurements: list[Measurement] = dataclasses.field(default_factory=list)


# --------------------------------------
#           PARSERS
# --------------------------------------


class ResultParser(abc.ABC):
    @abc.abstractmethod
    def parse(
        self, execution: Execution, stdout: str, stderr: str
    ) -> ExecutionResult: ...


class PlainSecondsParser(ResultParser):
    def parse(self, execution: Execution, stdout: str, stderr: str) -> ExecutionResult:
        result = ExecutionResult()

        for line in stdout.split("\n"):
            try:
                time = float(line) * 1000
                result.measurements.append(Measurement.runtime(execution, time, "ms"))
            except ValueError:
                pass

        return result


class LastLineParser(ResultParser):
    subparser: ResultParser

    def __init__(self, subparser: ResultParser) -> None:
        self.subparser = subparser

    def parse(self, execution: Execution, stdout: str, stderr: str) -> ExecutionResult:
        stdout_line = ""
        for stdout_line in reversed(stdout.split("\n")):
            if stdout_line.strip() != "":
                break

        stderr_line = ""
        for stderr_line in reversed(stderr.split("\n")):
            if stderr_line.strip() != "":
                break

        return self.subparser.parse(execution, stdout_line, stderr_line)


class RegexParser(ResultParser):
    type MatchGroup = str | int
    type OutputType = Literal["stdout", "stderr", "both"]

    metric: str
    regex: re.Pattern[str]
    output: OutputType

    match_group: MatchGroup
    process: Callable[[str], float]

    unit: Optional[str]
    unit_match_group: Optional[MatchGroup]

    iterations: bool

    def __init__(
        self,
        metric: str,
        regex: re.Pattern[str],
        output: OutputType,
        match_group: MatchGroup,
        process: Callable[[str], float] = float,
        unit: Optional[str] = None,
        unit_match_group: Optional[MatchGroup] = None,
        iterations: bool = False,
    ) -> None:
        self.metric = metric
        self.regex = regex
        self.output = output

        self.match_group = match_group
        self.process = process

        self.unit = unit
        self.unit_match_group = unit_match_group

        self.iterations = iterations

    def parse(self, execution: Execution, stdout: str, stderr: str) -> ExecutionResult:
        result = ExecutionResult()
        iteration = 1

        if self.output == "stdout":
            outputs = [stdout]
        elif self.output == "stderr":
            outputs = [stderr]
        elif self.output == "both":
            outputs = [stdout, stderr]
        else:
            raise ValueError(f"Unknown output type {self.output}")

        for output in outputs:
            pos = 0
            while (match := self.regex.search(output, pos)) is not None:
                pos = match.end()
                value = self.process(match.group(self.match_group))

                if self.iterations:
                    info = {"iteration": str(iteration)}
                    iteration += 1
                else:
                    info = {}

                if self.unit_match_group is not None:
                    unit = match.group(self.unit_match_group)
                elif self.unit is not None:
                    unit = self.unit
                else:
                    unit = ""

                result.measurements.append(
                    Measurement(
                        execution,
                        self.metric,
                        value,
                        unit,
                        info,
                    )
                )

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

    def parse(self, execution: Execution, stdout: str, stderr: str) -> ExecutionResult:
        result = ExecutionResult()
        iteration = 0

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

                result.measurements.append(
                    Measurement(
                        execution,
                        "runtime",
                        time,
                        "ms",
                        {"iteration": str(iteration)},
                    )
                )
                iteration += 1
                continue

            match = self.re_extra_criterion_log_line.match(line)
            if match is not None:
                # Match groups
                value = float(match.group("value"))
                unit = match.group("unit")
                criterion = match.group("criterion")

                # Add measurement
                result.measurements.append(
                    Measurement(
                        execution,
                        criterion,
                        value,
                        unit,
                        {"iteration": str(iteration)},
                    )
                )

                # Force new iteration
                if criterion == "total":
                    iteration += 1
                continue

        return result


class MixedResultParser(ResultParser):
    parsers: list[ResultParser]

    def __init__(self, *parsers: ResultParser) -> None:
        self.parsers = list(parsers)

    def parse(self, execution: Execution, stdout: str, stderr: str) -> ExecutionResult:
        result = ExecutionResult()

        for parser in self.parsers:
            result.measurements += parser.parse(execution, stdout, stderr).measurements

        return result


def time_parser(columns: list[TimeBinutilColumns]) -> ResultParser:
    parsers = []

    def mk_parser(column_name: str, unit: str) -> RegexParser:
        return RegexParser(
            column_name,
            re.compile(
                rf"^{column_name}: (\d+\.?\d*)$",
                re.MULTILINE,
            ),
            "stderr",
            match_group=1,
            unit=unit,
        )

    for col in columns:
        if col == "maximum_resident_size":
            parsers.append(mk_parser("maximum_resident_size", "kB"))

        elif col == "average_resident_size":
            parsers.append(mk_parser("average_resident_size", "kB"))

        elif col == "user_time":
            parsers.append(mk_parser("user_time", "s"))

        elif col == "system_time":
            parsers.append(mk_parser("system_time", "s"))

        elif col == "clock_time":
            parsers.append(mk_parser("clock_time", "s"))

        else:
            raise ValueError(f"Unknown Time column {col}")

    return MixedResultParser(*parsers)


# --------------------------------------
#           TUI
# --------------------------------------


class TUI:
    if sys.stdout.isatty():
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
#           REPORTERS
# --------------------------------------


class Reporter(abc.ABC):
    @abc.abstractmethod
    def report(self, result: ExecutionResult): ...

    @staticmethod
    def metrics(result: ExecutionResult) -> list[str]:
        out = []

        for measure in result.measurements:
            if measure.metric not in out:
                out.append(measure.metric)

        return out

    @staticmethod
    def measurement_info_columns(result: ExecutionResult) -> list[str]:
        out = []

        for measure in result.measurements:
            for col in measure.measurement_info.keys():
                if col not in out:
                    out.append(col)

        return out

    @staticmethod
    def info_columns(result: ExecutionResult) -> list[str]:
        out = []

        for measure in result.measurements:
            for col in measure.execution.info.keys():
                if col not in out:
                    out.append(col)

        return out


class MixedReporter(Reporter):
    reporters: list[Reporter]

    def __init__(self, *reporters: Reporter) -> None:
        self.reporters = list(reporters)

    def report(self, result: ExecutionResult):
        for r in self.reporters:
            r.report(result)


class CsvReporter(Reporter):
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

    def report(self, result: ExecutionResult):
        info_cols = Reporter.info_columns(result)
        measurement_info_cols = Reporter.measurement_info_columns(result)

        columns = (
            ["benchmark", "suite"]
            + info_cols
            + measurement_info_cols
            + ["metric", "value", "unit"]
        )

        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(self.filepath, "wt") as file:
            file.write(self.format_line(columns))

            for measure in result.measurements:
                line: list[str] = [
                    measure.execution.benchmark_name,
                    measure.execution.suite,
                ]

                for col in info_cols:
                    line.append(measure.execution.info.get(col, ""))

                for col in measurement_info_cols:
                    line.append(measure.measurement_info.get(col, ""))

                line += [measure.metric, str(measure.value), measure.unit]

                file.write(self.format_line(line))


class TableReporter(Reporter):
    def report(self, result: ExecutionResult):
        info_cols = Reporter.info_columns(result)
        measurement_info_cols = Reporter.measurement_info_columns(result)
        metrics = Reporter.metrics(result)

        # Measure widths
        benchmark_col_w = len("benchmark")
        suite_col_w = len("suite")
        info_cols_w = {info: len(info) for info in info_cols}
        measurement_info_w = {info: len(info) for info in measurement_info_cols}
        metric_w = len("metric")
        value_w = len("value")
        unit_w = len("unit")

        for measure in result.measurements:
            benchmark_col_w = max(
                len(measure.execution.benchmark_name), benchmark_col_w
            )
            suite_col_w = max(len(measure.execution.suite), suite_col_w)

            for i in info_cols:
                info_cols_w[i] = max(
                    len(measure.execution.info.get(i, "")), info_cols_w[i]
                )

            for i in measurement_info_cols:
                measurement_info_w[i] = max(
                    len(measure.measurement_info.get(i, "")), measurement_info_w[i]
                )

            metric_w = max(len(measure.metric), metric_w)
            value_w = max(len(str(measure.value)), value_w)
            unit_w = max(len(measure.unit), unit_w)

        # Print header
        sep_size = sum(
            [
                benchmark_col_w + 2,
                suite_col_w + 2,
                sum(info_cols_w.values()),
                len(info_cols_w) * 2,
                sum(measurement_info_w.values()),
                len(measurement_info_w) * 2,
                metric_w + 2,
                value_w + 2,
                unit_w,
            ]
        )

        print("\n" + "-" * sep_size)
        print("benchmark".ljust(benchmark_col_w + 2), end="")
        print("suite".ljust(suite_col_w + 2), end="")

        for i in info_cols:
            print(i.ljust(info_cols_w[i] + 2), end="")

        for i in measurement_info_cols:
            print(i.ljust(measurement_info_w[i] + 2), end="")

        print("metric".ljust(metric_w + 2), end="")
        print("value".ljust(value_w + 2), end="")
        print("unit".ljust(unit_w), end="")
        print("\n" + "-" * sep_size)

        # Print
        for measure in result.measurements:
            print(measure.execution.benchmark_name.ljust(benchmark_col_w + 2), end="")
            print(measure.execution.suite.ljust(suite_col_w + 2), end="")

            for i in info_cols:
                print(
                    measure.execution.info.get(i, "").ljust(
                        info_cols_w[i] + 2,
                    ),
                    end="",
                )

            for i in measurement_info_cols:
                print(
                    measure.measurement_info.get(i, "").ljust(
                        measurement_info_w[i] + 2,
                    ),
                    end="",
                )

            print(measure.metric.ljust(metric_w + 2), end="")
            print(str(measure.value).ljust(value_w + 2), end="")
            print(measure.unit.ljust(unit_w + 2), end="")
            print()

        print("-" * sep_size)


# --------------------------------------
#           EXECUTORS
# --------------------------------------


class Executor(abc.ABC):
    @abc.abstractmethod
    def execute(self, execution: Execution): ...

    def execute_all(self, executions: list[Execution]) -> Optional[ExecutionResult]:
        for execution in executions:
            self.execute(execution)
        return None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class DefaultExecutor(Executor):
    """
    The main Executor, which executes given commands, reporting success or
    failures to reporter
    """

    all_executions: Optional[int]
    finished_executions: int
    failed_executions: int

    reporter: Reporter
    crash_folder: Path

    result: ExecutionResult

    def __init__(self, crash_folder: Path, reporter: Reporter) -> None:
        self.all_executions = None
        self.finished_executions = 0
        self.failed_executions = 0

        self.reporter = reporter
        self.crash_folder = crash_folder

        self.result = ExecutionResult()

    def execute_all(self, executions: list[Execution]) -> ExecutionResult:
        self.all_executions = len(executions)
        super().execute_all(executions)
        return self.result

    def execute(self, execution: Execution):
        cmd = shutil.which(execution.command[0])
        if cmd is None:
            self.error_execution(
                execution, f"Command not found ({execution.command[0]})"
            )
            return

        execution.command[0] = cmd

        self.start_execution(execution)

        proc = None
        try:
            # TODO: group, process group?
            proc = subprocess.run(
                execution.command,
                # run spec
                capture_output=True,
                check=False,
                # Popen spec
                stdin=None,
                shell=False,
                cwd=execution.working_directory,
                env=execution.env,
                text=True,
            )

            if proc.returncode != 0:
                self.error_execution(
                    execution,
                    f"Program ended with non-zero return code ({proc.returncode})",
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                )
                return

            self.finalize(
                execution,
                stdout=proc.stdout,
                stderr=proc.stderr,
            )

        except OSError as e:
            self.error_execution(execution, str(e))

    def start_execution(self, execution: Execution) -> None:
        print(
            "["
            + f"{TUI.RED}{TUI.BOLD}{self.failed_executions}{TUI.RESET}"
            + f"/{TUI.GREEN}{TUI.BOLD}{self.finished_executions}{TUI.RESET}"
            + (
                f"/{TUI.BLUE}{TUI.BOLD}{self.all_executions}{TUI.RESET}"
                if self.all_executions is not None
                else ""
            )
            + "] "
            + execution.as_identifier()
            + "\n",
            end="",
        )

    def error_execution(
        self,
        execution: Execution,
        msg: str,
        stdout: Optional[str] = None,
        stderr: Optional[str] = None,
    ):
        self.failed_executions += 1
        print(
            f"{TUI.RED}{TUI.BOLD}Error in {execution.as_identifier()}{TUI.RESET}\n{msg}\n"
        )

        if stdout is not None or stderr is not None:
            run_path = self.crash_folder / execution.as_identifier().replace(" ", "_")
            run_path.mkdir(parents=True, exist_ok=True)

            if stdout is not None:
                with open(run_path / "stdout", "wt") as file:
                    file.write(stdout)

            if stderr is not None:
                with open(run_path / "stderr", "wt") as file:
                    file.write(stderr)

            print(
                f"{TUI.RED}stdout and stderr are saved in {str(run_path)}{TUI.RESET}\n"
            )

    def finalize(self, execution: Execution, stdout: str, stderr: str) -> None:
        self.finished_executions += 1
        self.result.measurements.extend(
            execution.parser.parse(execution, stdout, stderr).measurements
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.reporter.report(self.result)
        return False


class ParallelExecutor(DefaultExecutor):
    pool: ThreadPoolExecutor
    lock: Lock
    in_process_runs: int
    last_info: Optional[str]

    def __init__(self, ncores: int, crash_folder: Path, reporter: Reporter) -> None:
        super().__init__(crash_folder, reporter)

        self.pool = ThreadPoolExecutor(max_workers=ncores)
        self.lock = Lock()
        self.in_process_runs = 0
        self.last_info = None

    def execute(self, execution: Execution):
        self.pool.submit(super().execute, execution)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.pool.shutdown(wait=True)
        super().__exit__(*args)
        return False

    def print_execution(self):
        assert self.last_info is not None

        print(
            "["
            + f"{TUI.MAGENTA}{TUI.BOLD}{self.in_process_runs}{TUI.RESET}"
            + f"/{TUI.RED}{TUI.BOLD}{self.failed_executions}{TUI.RESET}"
            + f"/{TUI.GREEN}{TUI.BOLD}{self.finished_executions}{TUI.RESET}"
            + (
                f"/{TUI.BLUE}{TUI.BOLD}{self.all_executions}{TUI.RESET}"
                if self.all_executions is not None
                else ""
            )
            + "] "
            + self.last_info
            + "\n",
            end="",
        )

    def start_execution(self, execution: Execution) -> None:
        with self.lock:
            self.in_process_runs += 1
            self.last_info = execution.as_identifier()
            self.print_execution()

    def error_execution(self, execution, msg, stdout=None, stderr=None):
        with self.lock:
            super().error_execution(execution, msg, stdout, stderr)

    def finalize(self, execution: Execution, stdout: str, stderr: str) -> None:
        with self.lock:
            self.in_process_runs -= 1
            super().finalize(execution, stdout, stderr)
            self.print_execution()


class DryExecutor(Executor):
    """
    Simple executor which only prints what would be executed
    """

    def execute(self, execution: Execution):
        print("Execution:", " ".join(execution.command))
        print("Working working_directory:", str(execution.working_directory))
        print("Environment: ", end="")
        pprint(execution.env)
        print("Info: ", end="")
        pprint(execution.info)
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


def parse_params(*params: str, **kwarg_params: Any) -> Parameters:
    """
    Create a default argument parser and run it on argv
    """
    parser = make_argparser(*params, **kwarg_params)
    args = parser.parse_args()
    return Parameters.from_namespace(args)


# --------------------------------------
#          DEFAULT MAIN
# --------------------------------------


def main(
    config: Config,
    *params: str,
    reporter: Optional[Reporter] = None,
    executor: Optional[Executor] = None,
    **kwarg_params: Any,
) -> Optional[ExecutionResult]:
    """
    Sane default main. config is the benchmarks configuration that will be
    executed, params is a list of required parameters from the user,
    kwarg_params are optional parameters with their default value.
    """
    parser = make_argparser(*params, **kwarg_params)

    defp = parser.add_argument_group("Default benchr parameters")
    if reporter is None or executor is None:
        defp.add_argument(
            "--output",
            help="Where to store the results (Default: ./output)",
            metavar="file",
            type=str,
            default="./output",
            dest="__output",
        )

    if executor is None:
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

    ps = Parameters.from_namespace(parser.parse_args())

    executions = list(config.get_executions(ps))

    output: Path = Path(ps.__output).resolve()
    if reporter is None:
        reporter = CsvReporter(output / "results.csv")

    if executor is None:
        if ps.__dry:
            executor = DryExecutor()

        crash_folder = output / "crash"
        reporter = MixedReporter(TableReporter(), reporter)

        if ps.__jobs > 1:
            executor = ParallelExecutor(ps.__jobs, crash_folder, reporter)
        else:
            executor = DefaultExecutor(crash_folder, reporter)

    try:
        with executor:
            return executor.execute_all(executions)
    except KeyboardInterrupt:
        print("Interrupted")
        sys.exit(0)


# TODO: Run info - date, reflog

# --------------------------------------
#           EXPORTS
# --------------------------------------

__all__ = [
    # Definitions
    "Env",
    "Command",
    "Parameters",
    "TimeBinutilColumns",
    # Input definitions
    "Execution",
    "Benchmark",
    "B",
    # Suites of benchmarks
    "Suite",
    "suite",
    # Suite decorators
    "SuiteDecorator",
    "MatrixSuite",
    "TimeSuite",
    # Configuration
    "Config",
    # Result definitions
    "Measurement",
    "ExecutionResult",
    # Parsers
    "ResultParser",
    "PlainSecondsParser",
    "LastLineParser",
    "RegexParser",
    "RebenchParser",
    "MixedResultParser",
    "time_parser",
    # Reporters
    "Reporter",
    "MixedReporter",
    "CsvReporter",
    "TableReporter",
    # Executors
    "Executor",
    "DefaultExecutor",
    "ParallelExecutor",
    "DryExecutor",
    # ArgumentParsing
    "make_argparser",
    "parse_params",
    # Default main
    "main",
    # Reexports
    "Path",
]
