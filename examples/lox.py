from benchr import *


class ZooBatchParser(ResultParser):
    def parse(self, process_result: ProcessResult) -> ExecutionResult:
        # The second number is the throughput, first is sum and third is time, always ~10s
        if isinstance(process_result, SuccesfulProcessResult):
            sum_seen = False
            for line in process_result.stdout.split("\n"):
                try:
                    value = float(line)
                    if sum_seen:
                        return ExecutionResult(
                            [
                                Measurement(
                                    execution=process_result.execution,
                                    metric="throughput",
                                    value=value,
                                    unit="iteration",
                                    measurement_info={},
                                )
                            ]
                        )
                    else:
                        sum_seen = True

                except ValueError:
                    pass

        return ExecutionResult()


conf = (
    Config(
        [
            suite(
                name="LoxSuite",
                benchmarks=lambda ps: list(
                    filter(
                        lambda b: b.name != "zoo_batch",
                        Benchmark.from_folder(ps.cwd / "benchmarks", extension="lox"),
                    )
                ),
                parser=(
                    LastLineParser(PlainFloatParser("s"))
                    & ResourceUsageParser("maximum_resident_size")
                    & ClockTimeParser()
                )
                .kind("LIB")
                .note_failure()
                .note_timeout(),
            ),
            suite(
                name="ZooBatch",
                benchmarks=lambda ps: [
                    B("zoo_batch", path=ps.cwd / "benchmarks" / "zoo_batch.lox")
                ],
                parser=ZooBatchParser().kind("HIB").note_failure(),
            ).timeout(12),
        ]
    )
    .runs(2)
    .working_directory(lambda ps, _: ps.cwd / "benchmarks")
    .command(
        lambda ps, bench: [
            ps.lox,
            str(bench.keys.path),
        ]
    )
)

if __name__ == "__main__":
    main(
        conf,
        ["lox"],
        {"cwd": Path(__file__).parent},
    )
