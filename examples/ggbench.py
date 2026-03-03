import benchr


(
    ggbench()
    .matrix_axis(env=E(USE_RCP=[0, 1]))
    .parse_stdout(time=ParserTemplate("* iteration: 1, time: %d ms"))
    .parse_stdout(coverage=Parser())
    .check("retval")  # implicit
    .output(csv="output.csv", json="output.json", stdout="outputs_stdouts")
    .command(lambda file: ["./lox", file])
    .compare("time")
    .compare("coverage", PER_OUTPUT)
)


def make(parameters: benchr.Parameters):
    class LoxSuite(benchr.Suite):
        parser = MixedParser(
            time=TemplateParser(STDOUT, "* iteration: 1, time: %d ms"),
            coverage=CoverageParser(STDOUT),
        )

        command = lambda benchmark: [
            "./lox",
            f"{benchmark.name}.lox",
        ]

        env = {"USE_RCP": str(parameters.use_rcp)}


if __name__ == "__main__":
    params = benchr.parse_params()

    for use_rcp in [0, 1]:
        config = benchr.Config(LoxSuite(), env={"USE_RCP": str(use_rcp)})


suite(
    bench()
)
