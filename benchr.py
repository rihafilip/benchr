from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TextIO

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
