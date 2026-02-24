from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Self, TextIO, Any

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


# TODO: parsing results
class BenchmarkRunner(Executor):
    result_path: Path
    result: TextIO
    csv_headers: list[str]

    def __init__(
        self, explicit: bool, result_path: Path, csv_headers: list[str]
    ) -> None:
        super().__init__(explicit)
        self.result_path = result_path
        self.csv_headers = csv_headers

    def execute(
        self,
        command: list[str],
        folder: Path,
        env: Env,
        info: dict[str, str],
    ): ...

    def __enter__(self):
        # TODO: Check rebench-denoise
        self.result = open(self.result_path, "w")
        self.result.__enter__()
        self.result.write(",".join(self.csv_headers))
        self.result.write("\n")
        return self

    def __exit__(self, *args):
        return self.result.__exit__(*args)


# --------------------------------------

CWD = Path(__file__)


Matrix = dict[str, list[str]]


class Parser: ...


@dataclass
class Benchmark:
    name: str

    command: list[str]
    folder: Path
    env: Env
    matrix: Matrix
    parser: Parser


@dataclass
class Suite:
    name: str
    benchmarks: list[Benchmark]


@dataclass
class Config:
    suites: list[Suite]


Variables = dict[str, str]
RecVariables = str | dict[str, "RecVariables"]


class ConfigDecodeError(Exception):
    message: str

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)

    def extend(self, message: str) -> "ConfigDecodeError":
        return ConfigDecodeError(f"{message}: {self.message}")


@dataclass
class IncompleteFields:
    command: Optional[list[str]] = None
    folder: Optional[Path] = None
    env: Optional[Env] = None
    matrix: Optional[Matrix] = None
    variables: Variables = dict()
    parser: Optional[Parser] = None


@dataclass
class IncompleteBenchmark:
    name: str
    incomplete_fields: IncompleteFields = IncompleteFields()


@dataclass
class IncompleteSuite:
    name: str
    benchmarks: list[IncompleteBenchmark] = list()
    incomplete_fields: IncompleteFields = IncompleteFields()


@dataclass
class IncompleteConfig:
    suites: list[IncompleteSuite] = list()
    parameters: list[str] = list()
    incomplete_fields: IncompleteFields = IncompleteFields()


def assert_decode_type(obj: Any, *types, msg: Optional[str] = None) -> None:
    if not isinstance(obj, *types):
        if len(types) > 1:
            t_msg = f"{', '.join(types)}"
        else:
            t_msg = f"{types}"

        raise ConfigDecodeError(
            f"{'' if msg is None else msg}: Incorrect type, expected {t_msg}, got {type(obj)}"
        )


def flatten_variables(vars: dict[str, RecVariables]) -> Variables:
    assert_decode_type(vars, dict)

    res: Variables = {}
    for k, v in vars.items():
        assert_decode_type(k, str)
        assert_decode_type(v, str, dict)

        if isinstance(v, str):
            res[k] = v
        else:
            v = flatten_variables(v)
            for k2, v2 in v.items():
                res[f"{k}.{k2}"] = v2

    return res


def decode_fields(obj) -> IncompleteFields:
    assert_decode_type(obj, dict)
    res = IncompleteFields()

    return res


def decode_benchmark(obj) -> IncompleteBenchmark: ...


def decode_suite(obj, name) -> IncompleteSuite:
    assert_decode_type(name, str)

    try:
        assert_decode_type(obj, dict)
        res = IncompleteSuite(name=name)

        if "benchmarks" not in obj:
            raise ConfigDecodeError("Missing benchmarks in suite")

        try:
            benchs = obj["benchmarks"]
            assert_decode_type(benchs, list)
            res.benchmarks = [decode_benchmark(bench) for bench in benchs]
        except ConfigDecodeError as e:
            raise e.extend("In benchmarks")

        return res

    except ConfigDecodeError as e:
        raise e.extend(f"In suite {name}")


def decode_config(obj) -> IncompleteConfig:
    assert_decode_type(obj, dict, msg="Top configuration")

    res = IncompleteConfig()

    if "suites" not in obj:
        raise ConfigDecodeError("Missing suites in configuration")

    suites = obj["suites"]
    try:
        assert_decode_type(suites, dict)

        res.suites = [decode_suite(suite, name) for suite, name in suites.items()]
    except ConfigDecodeError as e:
        raise e.extend("In suites")

    res.parameters = obj.get("parameters", list())
    assert_decode_type(res.parameters, list, msg="Parameters")

    if "default" in obj:
        try:
            res.incomplete_fields = decode_fields(obj["default"])
        except ConfigDecodeError as e:
            raise e.extend("In default")

    if "variables" in obj:
        try:
            res.incomplete_fields.variables |= flatten_variables(obj["variables"])
        except ConfigDecodeError as e:
            raise e.extend("In variables")

    return res


# Everything always propagates from bottom up (Config -> Suite -> Benchmark):
# Available at all levels (default, suites.<name>, suites.<name>.benchmarks.<idx>):
#   command - list of strings, required
#   parser - string, required
#   working_directory - string, defaults to {cwd}
#   env - dictionary of strings, defaults to empty
#   matrix - dictionary of lists of strings, defaults to empty
#   extra - dictionary of strings, defaults to empty
#
# Top level:
#   parameters - list of strings
#   variables - dictionary of strings
#   default - dictionary
#   suites - definition of suites, dictonary of `suite`
#
# `suite`:
#   benchmarks - list of `benchmark` or string
#
# `benchmark`:
#   name - string, required
#
# Config exceptions:
#   Benchmark only specified as a string is equal to dictionary of only the name
#
# Substitution:
#   "cwd" - working directory
#   "matrix" - dictionary of currently active matrix fields
#   "suite" - the current suite
#   "benchmark" - the current benchmark
#   everything specified in "variables"
#   "extra" either as a dictionary or single string

if __name__ == "__main__":
    print("Hello world")
