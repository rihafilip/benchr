import abc
import argparse
import dataclasses
import os
import re
import resource
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from threading import Lock
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Iterator, Literal, Optional, Sequence

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

    def __getitem__(self, name: str) -> Any:
        return getattr(self, name)

    @staticmethod
    def from_namespace(ns: argparse.Namespace) -> "Parameters":
        """
        Convert the argparse.Namespace to Parameters
        """
        return Parameters(**vars(ns))


@dataclass
class SuccesfulProcessResult:
    execution: "Execution"
    runtime: float
    stdout: str
    stderr: str
    rusage: Optional[resource.struct_rusage]


@dataclass
class FailedProcessResult:
    execution: "Execution"
    runtime: Optional[float]
    stdout: Optional[str]
    stderr: Optional[str]
    rusage: Optional[resource.struct_rusage]

    returncode: int
    reason: Literal["timed_out", "non_zero_returncode"] | str

    @staticmethod
    def empty(
        execution: "Execution",
        reason: Literal["timed_out", "non_zero_returncode"] | str,
    ) -> "FailedProcessResult":
        return FailedProcessResult(
            execution=execution,
            runtime=None,
            stdout=None,
            stderr=None,
            rusage=None,
            returncode=0,
            reason=reason,
        )


ProcessResult = SuccesfulProcessResult | FailedProcessResult

ResourceMetric = Literal[
    "maximum_resident_size",
    "user_time",
    "system_time",
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

    timeout: Optional[float]

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
        """
        A intermediate state of Execution
        """

        benchmark_name: str
        data: tuple[Any, ...] | Any
        keys: SimpleNamespace

        suite: str
        parser: Optional["ResultParser"]

        command: Optional[Command]
        working_directory: Optional[Path]
        env: Env

        timeout: Optional[float]

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
                timeout=self.timeout,
                info=self.info,
            )


@dataclass
class Benchmark:
    """
    A definition of one benchmark. data and keys can be any benchmark-specific
    data that are needed for its execution.

    `data` is specified by positional arguments. If there is a single argument,
    `data` will not be a tuple but just that argument.

    `keys` are specified by keyword arguments
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
        """
        Create benchmarks from Path, where the name will be the filename
        without extension, and `keys.path` is the full given path
        """
        return [
            Benchmark(
                file.stem,
                path=file,
            )
            for file in files
        ]

    @staticmethod
    def from_folder(folder: Path, extension: Optional[str] = None) -> list["Benchmark"]:
        """
        Recursively walk the given folder, collecting all files with the given
        extension (or all if no extension is given) into Benchmarks
        """
        res = []
        for path, _, files in folder.walk():
            for file in files:
                p = path / file
                if extension is None or p.suffix.lower() == ("." + extension.lower()):
                    res.append(p)

        return Benchmark.from_files(*res)


B = Benchmark

# --------------------------------------
#          SUITES OF BENCHMARKS
# --------------------------------------


class BenchmarkCollection[This](abc.ABC):
    """
    Abstract superclass of Suite and Benchmark - implementation detail
    """

    @abc.abstractmethod
    def apply_suite_decorator(self, decorator: Callable[["Suite"], "Suite"]) -> This:
        """
        Apply a decorator to all Suites inside this collection
        """
        ...

    def matrix[T](
        self,
        matrix: "Matrix[T]",
    ) -> This:
        """
        Add a matrix parameter. Each benchmark is going to be duplicated with
        each instance having one value from `parameters`.

        If `working_directory` is not None, it will also change the working
        directory of each benchmark.

        If `env` is not None, it will add a new environment variables to the
        benchmarks environment.
        """
        return self.apply_suite_decorator(matrix.build)

    def runs(self, value: int) -> This:
        """
        Run each benchmark `value` times without any other modification
        """
        return self.matrix(Matrix("run", range(1, value + 1)))

    def timeout(self, timeout: float) -> This:
        """
        Set the timeout of benchmarks (in seconds)
        """
        return self.apply_suite_decorator(
            lambda suite: TimeoutSuite(
                suite,
                timeout,
            )
        )


class Suite(BenchmarkCollection["Suite"]):
    """
    A collection of benchmarks
    """

    def apply_suite_decorator(self, decorator: Callable[["Suite"], "Suite"]) -> "Suite":
        return decorator(self)

    @abc.abstractmethod
    def get_executions(
        self, parameters: Parameters
    ) -> Iterator[Execution.Incomplete]: ...

    def to_config(self) -> "Config":
        """
        Create a simplified config with only one Suite
        """
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
                data=b.data,
                keys=b.keys,
                suite=self.name,
                command=command,
                parser=self.parser,
                working_directory=working_directory,
                env=env,
                timeout=None,
                info={},
            )


def suite(
    name: str,
    benchmarks: Sequence[Benchmark | str] | Callable[[Parameters], list[Benchmark]],
    *,
    command: Optional[Callable[[Parameters, Benchmark], Command]] = None,
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
    """
    A Suite that extends another Suite
    """

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
    ) -> Iterator[Execution.Incomplete]:
        """
        This method needs to be implemented, as it is the one that extends the
        parent suite
        """
        ...


type MatrixCallable[T, R] = Callable[[Parameters, Execution.Incomplete, T], R]


class MatrixSuite[T](SuiteDecorator):
    name: str
    parameters: Sequence[T]

    matrix_command: Optional[MatrixCallable[T, Command]]
    matrix_working_directory: Optional[MatrixCallable[T, Path]]
    matrix_env: MatrixCallable[T, Env]
    matrix_info: Optional[Callable[[T], dict[str, str]]]

    def __init__(
        self,
        name: str,
        parent: Suite,
        parameters: Sequence[T],
        matrix_command: Optional[MatrixCallable[T, Command]],
        matrix_working_directory: Optional[MatrixCallable[T, Path]],
        matrix_env: MatrixCallable[T, Env],
        matrix_info: Optional[Callable[[T], dict[str, str]]],
    ) -> None:
        super().__init__(parent)

        self.name = name
        self.parameters = parameters

        self.matrix_command = matrix_command
        self.matrix_working_directory = matrix_working_directory
        self.matrix_env = matrix_env
        self.matrix_info = matrix_info

    def extend_execution(
        self, parameters: Parameters, execution: Execution.Incomplete
    ) -> Iterator[Execution.Incomplete]:
        for p in self.parameters:
            if self.matrix_command is not None:
                c = self.matrix_command(parameters, execution, p)
            else:
                c = execution.command

            e = execution.env | self.matrix_env(parameters, execution, p)

            if self.matrix_working_directory is not None:
                wd = self.matrix_working_directory(parameters, execution, p)
            else:
                wd = execution.working_directory

            if self.matrix_info is not None:
                i = execution.info | self.matrix_info(p)
            else:
                i = execution.info | {self.name: str(p)}

            yield dataclasses.replace(
                execution,
                command=c,
                env=e,
                working_directory=wd,
                info=i,
            )


@dataclass
class Matrix[T]:
    """
    The MatrixSuite builder
    """

    name: str
    parameters: Sequence[T]

    matrix_command: Optional[MatrixCallable[T, Command]] = None
    matrix_working_directory: Optional[MatrixCallable[T, Path]] = None
    matrix_env: MatrixCallable[T, Env] = const({})
    matrix_info: Optional[Callable[[T], dict[str, str]]] = None

    def __init__(
        self,
        name: str,
        parameters: Sequence[T],
    ) -> None:
        self.name = name
        self.parameters = parameters

    def command(self, callback: Callable[[T], Command]):
        return self.command_full(lambda ps, ex, p: callback(p))

    def command_full(self, callback: MatrixCallable[T, Command]):
        if self.matrix_command is not None:
            raise ValueError("Multiple definitions of command")

        return dataclasses.replace(self, matrix_command=callback)

    def working_directory(self, callback: Callable[[T], Path]):
        return self.working_directory_full(lambda ps, ex, p: callback(p))

    def working_directory_full(self, callback: MatrixCallable[T, Path]):
        if self.matrix_working_directory is not None:
            raise ValueError("Multiple definitions of working directory")

        return dataclasses.replace(self, matrix_working_directory=callback)

    def env(self, name: Optional[str]):
        if name is None:
            name = self.name

        return self.env_callback_full(lambda ps, ex, p: {name: str(p)})

    def env_callback(self, callback: Callable[[T], Env]):
        return self.env_callback_full(lambda ps, ex, p: callback(p))

    def env_callback_full(self, callback: MatrixCallable[T, Env]):
        prev_mk_env = self.matrix_env
        mk_env = lambda ps, ex, p: prev_mk_env(ps, ex, p) | callback(ps, ex, p)

        return dataclasses.replace(self, matrix_env=mk_env)

    def info(self, callback: Callable[[T], dict[str, str]]):
        if self.matrix_info is not None:
            raise ValueError("Multiple definitions of info")

        return dataclasses.replace(self, matrix_info=callback)

    def build(self, suite: Suite) -> MatrixSuite:
        return MatrixSuite(
            name=self.name,
            parent=suite,
            parameters=self.parameters,
            matrix_command=self.matrix_command,
            matrix_working_directory=self.matrix_working_directory,
            matrix_env=self.matrix_env,
            matrix_info=self.matrix_info,
        )


class TimeoutSuite(SuiteDecorator):
    timeout_value: float

    def __init__(
        self,
        parent: Suite,
        timeout_value: float,
    ) -> None:
        super().__init__(parent)
        self.timeout_value = timeout_value

    def extend_execution(
        self, parameters: Parameters, execution: Execution.Incomplete
    ) -> Iterator[Execution.Incomplete]:
        execution.timeout = self.timeout_value
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

    def parser(self, default_parser: "ResultParser") -> "Config":
        if self.default_parser is not None:
            raise ValueError("Multiple definitions of default parser")

        return dataclasses.replace(self, default_parser=default_parser)

    def command(
        self,
        default_command: Callable[[Parameters, Execution.Incomplete], Command]
        | Command,
    ) -> "Config":
        """
        Define a default command for all benchmarks
        """
        if self.default_command is not None:
            raise ValueError("Multiple definitions of default command")

        if not callable(default_command):
            default_command = const(default_command)

        return dataclasses.replace(self, default_command=default_command)

    def working_directory(
        self,
        default_working_directory: Callable[[Parameters, Execution.Incomplete], Path]
        | Path,
    ) -> "Config":
        """
        Define a default working directory for all benchmarks
        """
        if self.default_working_directory is not None:
            raise ValueError("Multiple definitions of default working directory")

        if not callable(default_working_directory):
            default_working_directory = const(default_working_directory)

        return dataclasses.replace(
            self, default_working_directory=default_working_directory
        )

    def env(
        self,
        default_env: Callable[[Parameters, Execution.Incomplete], Env] | Env,
    ) -> "Config":
        """
        Define a default environment for all benchmarks
        """
        if not callable(default_env):
            default_env = const(default_env)

        if self.default_env is not None:
            prev_default_env = self.default_env
            callback = lambda ps, e: prev_default_env(ps, e) | default_env(ps, e)
        else:
            callback = default_env

        return dataclasses.replace(self, default_env=callback)

    def get_executions(self, parameters: Parameters) -> list[Execution]:
        """
        Return all executions for this configuration
        """
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

                res.append(exe.finalize())
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

    def info_columns(self) -> list[str]:
        """
        Get all info categories on all Executions
        """
        out = []

        for measure in self.measurements:
            for col in measure.execution.info.keys():
                if col not in out:
                    out.append(col)

        return out

    def measurement_info_columns(self) -> list[str]:
        """
        Get all info categories on all measurements
        """
        out = []

        for measure in self.measurements:
            for col in measure.measurement_info.keys():
                if col not in out:
                    out.append(col)

        return out

    def metrics(self) -> list[str]:
        """
        Get all metrics in the result
        """
        out = []

        for measure in self.measurements:
            if measure.metric not in out:
                out.append(measure.metric)

        return out

    def to_data_frame(self, pivoted: bool = False, units: Optional[bool] = None):
        import pandas as pd

        if units is None:
            units = not pivoted

        info_cols = self.info_columns()
        measurement_info_cols = self.measurement_info_columns()

        rows = []
        for m in self.measurements:
            row: dict[str, Any] = {
                "benchmark": m.execution.benchmark_name,
                "suite": m.execution.suite,
            }

            for col in info_cols:
                row[col] = m.execution.info.get(col, "")

            for col in measurement_info_cols:
                row[col] = m.measurement_info.get(col, "")

            if pivoted:
                row[m.metric] = m.value
                if units:
                    row[m.metric + "_unit"] = m.unit
            else:
                row["metric"] = m.metric
                row["value"] = m.value
                if units:
                    row["unit"] = m.unit

            rows.append(row)

        index_cols = ["benchmark", "suite"] + info_cols + measurement_info_cols

        df = pd.DataFrame(rows)
        df.set_index(index_cols, inplace=True)
        return df


# --------------------------------------
#           PARSERS
# --------------------------------------


class ResultParser(abc.ABC):
    """
    Parse stdout and stderr into results
    """

    @abc.abstractmethod
    def parse(self, process_result: ProcessResult) -> ExecutionResult: ...

    def ignore_fail(self) -> "ResultParser":
        """
        Ignore failed executions, parsing them as succesful
        """
        return IgnoreFailParserDecorator(self)

    def note_failure(self, dummy_metric: str = "runtime") -> "ResultParser":
        """
        Add a column `failed` with values 1 and 0
        """
        return NoteFailureParserDecorator(self, dummy_metric)

    def note_timeout(self, runtime_metric: str = "runtime") -> "ResultParser":
        """
        Add a column `timed_out` with values 1 and 0
        """
        return NoteTimeoutParserDecorator(self, runtime_metric)

    def kind(
        self, measure_kind: "MeasurementKindParserDecorator.Kind"
    ) -> "ResultParser":
        return MeasurementKindParserDecorator(self, measure_kind)

    def __and__(self, other) -> "ResultParser":
        return MixedResultParser(self, other)


class MixedResultParser(ResultParser):
    """
    Multiple parsers posing as one
    """

    parsers: list[ResultParser]

    @staticmethod
    def canonize(parsers: Iterable[ResultParser]) -> Iterator[ResultParser]:
        """
        Flatten the representation of MixedResultParser (one in another)
        """
        for parser in parsers:
            if isinstance(parser, MixedResultParser):
                for subparser in parser.parsers:
                    yield subparser
            else:
                yield parser

    def __init__(self, *parsers: ResultParser) -> None:
        self.parsers = list(canon_p for canon_p in MixedResultParser.canonize(parsers))

    def parse(self, process_result: ProcessResult) -> ExecutionResult:
        result = ExecutionResult()

        for parser in self.parsers:
            result.measurements += parser.parse(process_result).measurements

        return result


class PlainFloatParser(ResultParser):
    """
    Try to parse simple floats on each line as seconds. Only on succesful
    runs.
    """

    unit: str

    def __init__(self, unit: str) -> None:
        self.unit = unit

    def parse(self, process_result: ProcessResult) -> ExecutionResult:
        if isinstance(process_result, FailedProcessResult):
            return ExecutionResult()

        result = ExecutionResult()

        for line in process_result.stdout.split("\n"):
            try:
                time = float(line)
                result.measurements.append(
                    Measurement.runtime(process_result.execution, time, self.unit)
                )
            except ValueError:
                pass

        return result


class LastLineParser(ResultParser):
    """
    Only parse the last non-empty line of succesful runs
    """

    subparser: ResultParser

    def __init__(self, subparser: ResultParser) -> None:
        self.subparser = subparser

    def parse(self, process_result: ProcessResult) -> ExecutionResult:
        if isinstance(process_result, FailedProcessResult):
            return ExecutionResult()

        stdout_line = ""
        for stdout_line in reversed(process_result.stdout.split("\n")):
            if stdout_line.strip() != "":
                break

        stderr_line = ""
        for stderr_line in reversed(process_result.stderr.split("\n")):
            if stderr_line.strip() != "":
                break

        return self.subparser.parse(
            dataclasses.replace(
                process_result,
                stdout=stdout_line,
                stderr=stderr_line,
            )
        )


class RegexParser(ResultParser):
    """
    Parse the output of a succesful run based on a regex
    """

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

        if unit is None and unit_match_group is None:
            raise ValueError("Missing unit specification")
        self.unit = unit
        self.unit_match_group = unit_match_group

        self.iterations = iterations

    def parse(self, process_result: ProcessResult) -> ExecutionResult:
        if isinstance(process_result, FailedProcessResult):
            return ExecutionResult()

        result = ExecutionResult()
        iteration = 1

        if self.output == "stdout":
            outputs = [process_result.stdout]
        elif self.output == "stderr":
            outputs = [process_result.stderr]
        elif self.output == "both":
            outputs = [process_result.stdout, process_result.stderr]
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
                        process_result.execution,
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

    def parse(self, process_result: ProcessResult) -> ExecutionResult:
        if process_result.stdout is None:
            return ExecutionResult()

        result = ExecutionResult()
        iteration = 0

        for line in process_result.stdout.split("\n"):
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
                        process_result.execution,
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
                        process_result.execution,
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


class ClockTimeParser(ResultParser):
    """
    Capture the outer runtime of a process
    """

    def parse(self, process_result: ProcessResult) -> ExecutionResult:
        if process_result.runtime is None:
            return ExecutionResult()

        return ExecutionResult(
            [
                Measurement(
                    execution=process_result.execution,
                    metric="clock_time",
                    value=process_result.runtime,
                    unit="s",
                    measurement_info={},
                )
            ]
        )


class ResourceUsageParser(ResultParser):
    RUsageField = Literal[
        "ru_utime",
        "ru_stime",
        "ru_maxrss",
        "ru_ixrss",
        "ru_idrss",
        "ru_isrss",
        "ru_minflt",
        "ru_majflt",
        "ru_nswap",
        "ru_inblock",
        "ru_oublock",
        "ru_msgsnd",
        "ru_msgrcv",
        "ru_nsignals",
        "ru_nvcsw",
        "ru_nivcsw",
    ]

    field: RUsageField
    metric: str
    unit: str

    def __init__(self, field: RUsageField, metric: str, unit: str) -> None:
        self.field = field
        self.metric = metric
        self.unit = unit

    def parse(self, process_result: ProcessResult) -> ExecutionResult:
        if process_result.rusage is None:
            return ExecutionResult()

        value = getattr(process_result.rusage, self.field)

        # MacOS reports in B, not kB
        if sys.platform == "darwin" and self.field == "ru_maxrss":
            value /= 1024

        return ExecutionResult(
            [
                Measurement(
                    execution=process_result.execution,
                    metric=self.metric,
                    value=value,
                    unit=self.unit,
                    measurement_info={},
                )
            ]
        )


def resource_usage_parser(*columns: ResourceMetric) -> ResultParser:
    """
    Create a parser for resourece metrics
    """
    parsers = []

    for col in columns:
        if col == "maximum_resident_size":
            parsers.append(
                ResourceUsageParser(
                    "ru_maxrss",
                    "maximum_resident_size",
                    "kB",
                )
            )

        elif col == "user_time":
            parsers.append(
                ResourceUsageParser(
                    "ru_utime",
                    "user_time",
                    "s",
                )
            )

        elif col == "system_time":
            parsers.append(
                ResourceUsageParser(
                    "ru_stime",
                    "system_time",
                    "s",
                )
            )

        else:
            raise ValueError(f"Unknown resource metric column {col}")

    if len(parsers) == 1:
        return parsers[0]

    return MixedResultParser(*parsers)


# --------------------------------------
#           PARSER DECORATORS
# --------------------------------------


class IgnoreFailParserDecorator(ResultParser):
    subparser: ResultParser

    def __init__(self, subparser: ResultParser) -> None:
        self.subparser = subparser

    def parse(self, process_result: ProcessResult) -> ExecutionResult:
        if isinstance(process_result, FailedProcessResult):
            process_result = SuccesfulProcessResult(
                execution=process_result.execution,
                runtime=process_result.runtime or -1,
                stdout=process_result.stdout or "",
                stderr=process_result.stderr or "",
                rusage=process_result.rusage,
            )

        return self.subparser.parse(process_result)


def add_measurement_note(
    result: ExecutionResult, info: dict[str, str], dummy_measurement: Measurement
) -> ExecutionResult:
    if len(result.measurements) == 0:
        result.measurements.append(dummy_measurement)
    else:
        for m in result.measurements:
            m.measurement_info |= info

    return result


class NoteFailureParserDecorator(ResultParser):
    subparser: ResultParser
    dummy_metric: str

    def __init__(self, subparser: ResultParser, dummy_metric: str = "runtime") -> None:
        self.subparser = subparser
        self.dummy_metric = dummy_metric

    def parse(self, process_result: ProcessResult) -> ExecutionResult:
        result = self.subparser.parse(process_result)

        failed = 1 if isinstance(process_result, FailedProcessResult) else 0

        return add_measurement_note(
            result,
            {"failed": str(failed)},
            Measurement(
                execution=process_result.execution,
                metric=self.dummy_metric,
                value=0,
                unit="",
                measurement_info={"failed": str(failed)},
            ),
        )


class NoteTimeoutParserDecorator(ResultParser):
    subparser: ResultParser
    runtime_metric: str

    def __init__(
        self, subparser: ResultParser, runtime_metric: str = "runtime"
    ) -> None:
        self.subparser = subparser
        self.runtime_metric = runtime_metric

    def parse(self, process_result: ProcessResult) -> ExecutionResult:
        result = self.subparser.parse(process_result)

        timed_out = (
            1
            if isinstance(process_result, FailedProcessResult)
            and process_result.reason == "timed_out"
            else 0
        )

        dummy_measure = Measurement(
            execution=process_result.execution,
            metric=self.runtime_metric,
            value=process_result.execution.timeout or 0,
            unit="s",
            measurement_info={"timed_out": str(timed_out)},
        )

        result = add_measurement_note(
            result, {"timed_out": str(timed_out)}, dummy_measure
        )

        # If there is no measurement of the `runtime_metric`, add the dummy_measure
        if not any(map(lambda m: m.metric == self.runtime_metric, result.measurements)):
            result.measurements.append(dummy_measure)

        return result


class MeasurementKindParserDecorator(ResultParser):
    Kind = Literal["LIB", "HIB"]  # Lower / Higher is Better

    subparser: ResultParser
    measure_kind: Kind

    def __init__(self, subparser: ResultParser, measure_kind: Kind) -> None:
        self.subparser = subparser
        self.measure_kind = measure_kind

    def parse(self, process_result: ProcessResult) -> ExecutionResult:
        result = self.subparser.parse(process_result)

        for m in result.measurements:
            m.measurement_info["kind"] = self.measure_kind

        return result


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
    """
    Reports the results
    """

    @abc.abstractmethod
    def report(self, result: ExecutionResult): ...


class MixedReporter(Reporter):
    """
    Multiple reporters posing as one
    """

    reporters: list[Reporter]

    def __init__(self, *reporters: Reporter) -> None:
        self.reporters = list(reporters)

    def report(self, result: ExecutionResult):
        for r in self.reporters:
            r.report(result)


class CsvReporter(Reporter):
    """
    Report into CSV file
    """

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
        info_cols = result.info_columns()
        measurement_info_cols = result.measurement_info_columns()

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
    """
    Report into CLI
    """

    def report(self, result: ExecutionResult):
        info_cols = result.info_columns()
        measurement_info_cols = result.measurement_info_columns()

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
    """
    Execute the executions
    """

    @abc.abstractmethod
    def execute(self, execution: Execution):
        """
        Run single execution - implementation detail, use `execute_all`
        """
        ...

    def execute_all(self, executions: list[Execution]) -> Optional[ExecutionResult]:
        """
        Run all executions - this is the prefered way of running executions
        """
        for execution in executions:
            self.execute(execution)
        return None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class DefaultExecutor(Executor):
    """
    The main Executor
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
                FailedProcessResult.empty(
                    execution=execution,
                    reason=f"Command not found ({execution.command[0]})",
                )
            )
            return

        execution.command[0] = cmd

        self.start_execution(execution)

        try:
            proc = subprocess.Popen(
                execution.command,
                cwd=execution.working_directory,
                env=execution.env,
                stdin=None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=False,
            )
            starttime = time.monotonic()

            rusage = None
            timed_out = False
            if execution.timeout is not None:
                stoptime = time.monotonic() + execution.timeout

                # Busy wait
                while True:
                    try:
                        pid, _, rusage = os.wait4(proc.pid, os.WNOHANG)
                        assert pid == proc.pid or pid == 0
                    except ChildProcessError:
                        break

                    if pid == proc.pid:
                        break

                    if stoptime - time.monotonic() <= 0:
                        timed_out = True
            else:
                try:
                    _, _, rusage = os.wait4(proc.pid, 0)
                except ChildProcessError:
                    ...

            endtime = time.monotonic()
            runtime = endtime - starttime
            stdout, stderr = proc.communicate()

            if timed_out or proc.returncode != 0:
                result = FailedProcessResult(
                    execution=execution,
                    runtime=runtime,
                    stdout=stdout,
                    stderr=stderr,
                    rusage=rusage,
                    returncode=proc.returncode,
                    reason="timed_out" if timed_out else "non_zero_returncode",
                )
                self.error_execution(result)
            else:
                result = SuccesfulProcessResult(
                    execution=execution,
                    runtime=runtime,
                    stdout=stdout,
                    stderr=stderr,
                    rusage=rusage,
                )

            self.finalize(result)

        except OSError as e:
            self.error_execution(
                FailedProcessResult.empty(
                    execution=execution,
                    reason=str(e),
                )
            )

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

    def error_execution(self, process_result: FailedProcessResult):
        self.failed_executions += 1
        print(
            f"{TUI.RED}{TUI.BOLD}Error in {process_result.execution.as_identifier()}{TUI.RESET}\n"
        )

        if process_result.reason == "non_zero_returncode":
            print(
                f"Program ended with non-zero return code ({process_result.returncode})\n"
            )
        elif process_result.reason == "timed_out":
            print(
                f"Program timed out after {process_result.execution.timeout} seconds\n"
            )
        else:
            print(process_result.reason + "\n")

        if process_result.stdout is not None or process_result.stderr is not None:
            run_path = (
                self.crash_folder
                / process_result.execution.as_identifier().replace(" ", "_")
            )
            run_path.mkdir(parents=True, exist_ok=True)

            if process_result.stdout is not None:
                with open(run_path / "stdout", "wt") as file:
                    file.write(process_result.stdout)

            if process_result.stderr is not None:
                with open(run_path / "stderr", "wt") as file:
                    file.write(process_result.stderr)

            print(
                f"{TUI.RED}stdout and stderr are saved in {str(run_path)}{TUI.RESET}\n"
            )

    def finalize(self, process_result: ProcessResult) -> None:
        self.finished_executions += 1
        self.result.measurements.extend(
            process_result.execution.parser.parse(process_result).measurements
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.reporter.report(self.result)
        return False


class ParallelExecutor(DefaultExecutor):
    """
    An executor that runs multiple tasks in parallel - mostly usable for
    collecting metrics other that runtime
    """

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

    def error_execution(self, process_result: FailedProcessResult):
        with self.lock:
            super().error_execution(process_result)

    def finalize(self, process_result: ProcessResult) -> None:
        with self.lock:
            self.in_process_runs -= 1
            super().finalize(process_result)
            self.print_execution()


class DryExecutor(Executor):
    """
    Pseudo-executor which only prints the execution plan
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
    params: list[str],
    kwarg_params: dict[str, Any],
    reporter: Optional[Reporter] = None,
    executor: Optional[Executor] = None,
    output_folder: Optional[Path] = None,
) -> Optional[ExecutionResult]:
    """
    Sane default main. config is the benchmarks configuration that will be
    executed, params is a list of required parameters from the user,
    kwarg_params are optional parameters with their default value.

    If no reporter is specified, it defaults to CSV and CLI report.

    If no executor is specified, it defaults to DefaultExecutor, which can be
    changed with CLI arguments --dry and --jobs/-j.

    If no output_folder is specified, it defaults to ./output, which can be
    changed with CLI argument --output.
    """
    parser = make_argparser(*params, **kwarg_params)

    defp = parser.add_argument_group("Default benchr parameters")
    if (reporter is None or executor is None) and output_folder is None:
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

    if executor is None:
        if ps.__dry:
            executor = DryExecutor()
        else:
            if output_folder is None:
                output_folder = Path(ps.__output).resolve()

            if reporter is None:
                reporter = MixedReporter(
                    TableReporter(), CsvReporter(output_folder / "results.csv")
                )

            crash_folder = output_folder / "crash"

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
    "ProcessResult",
    "SuccesfulProcessResult",
    "FailedProcessResult",
    "ResourceMetric",
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
    "Matrix",
    "TimeoutSuite",
    # Configuration
    "Config",
    # Result definitions
    "Measurement",
    "ExecutionResult",
    # Parsers
    "ResultParser",
    "MixedResultParser",
    "PlainFloatParser",
    "LastLineParser",
    "RegexParser",
    "RebenchParser",
    "ClockTimeParser",
    "ResourceUsageParser",
    "resource_usage_parser",
    # Parser decorators
    "IgnoreFailParserDecorator",
    "NoteFailureParserDecorator",
    "NoteTimeoutParserDecorator",
    "MeasurementKindParserDecorator",
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
