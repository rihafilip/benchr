from benchr import *

conf = (
    suite(
        name="LoxSuite",
        working_directory=lambda ps, _: ps.cwd / "benchmarks",
        parser=LastLineParser(PlainSecondsParser()),
        command=lambda ps, bench: [
            ps.lox,
            str(bench.keys.path),
        ],
        benchmarks=lambda ps: Benchmark.from_folder(
            ps.cwd / "benchmarks", extension="lox"
        ),
    )
    .runs(2)
    .time(
        "maximum_resident_size",
        "average_resident_size",
        "user_time",
        "system_time",
        "clock_time",
    )
    .to_config()
)


if __name__ == "__main__":
    main(conf, ["lox"], {"cwd": Path(__file__).parent})
