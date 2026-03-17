import numpy as np
import pandas as pd
from benchr import *

pd.options.display.float_format = lambda x: f"{int(x)}" if x == int(x) else f"{x:.2f}"

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
    # .runs(5)
    .runs(2)
    .time(
        "maximum_resident_size",
        "clock_time",
    )
    .to_config()
)


if __name__ == "__main__":
    results = main(
        conf,
        ["lox"],
        {
            "cwd": Path(__file__).parent,
        },
    )
    assert results is not None

    df = results.to_data_frame()

    df["value"] = np.where(
        df["metric"] == "clock_time", df["value"] * 1000, df["value"]
    )
    df.drop("unit", axis=1, inplace=True)
    df = df.pivot(columns="metric")
    df.columns = df.columns.droplevel(0)
    df = df.groupby(["benchmark", "suite"]).mean()
    df.reset_index(inplace=True)
    print(df)

    df.to_csv("./output/pd.csv", index=False)
