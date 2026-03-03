from pathlib import Path

import benchr
from benchr import B, Config, Suite

conf = Config(
    Suite(
        name="LoxSuite",
        working_directory=lambda ps, _: ps.cwd / "benchmarks",
        command=lambda params, bench: [
            params.lox,
            f"{bench.name}.lox",
        ],
        benchmarks=[
            B("binary_trees"),
            B("equality"),
            B("fib"),
            B("instantiation"),
            B("invocation"),
            B("method_call"),
            B("properties"),
            B("string_equality"),
            B("zoo_batch"),
            B("zoo"),
        ],
    )
)

if __name__ == "__main__":
    benchr.main(conf, "lox", cwd=Path(__file__).parent)
