from benchr import *

INPUTS = Path(__file__).resolve() / "inputs"
BENCHMARKS = INPUTS / "Benchmarks"


LOCALE = {
    "LC_CTYPE": "en_US.UTF-8",
    "LC_TIME": "en_US.UTF-8",
    "LC_MONETARY": "en_US.UTF-8",
    "LC_PAPER": "en_US.UTF-8",
    "LC_ADDRESS": "C",
    "LC_MEASUREMENT": "en_US.UTF-8",
    "LC_NUMERIC": "C",
    "LC_COLLATE": "en_US.UTF-8",
    "LC_MESSAGES": "en_US.UTF-8",
    "LC_NAME": "C",
    "LC_TELEPHONE": "C",
    "LC_IDENTIFICATION": "C",
}

AreWeFast = suite(
    name="areWeFast",
    working_directory=BENCHMARKS / "areWeFast",
    benchmarks=[
        B("Mandelbrot", 500),
        B("Bounce", 35),
        B("Bounce_nonames", 35),
        B("Bounce_nonames_simple", 35),
        B("Storage", 100),
    ],
    parser=RebenchParser(),
    command=lambda parameters, benchmark: [
        str(Path(parameters.Rpath) / "bin" / "Rscript"),
        "harness.r",
        benchmark.name,
        str(parameters.iterations),
        str(benchmark.data),
    ],
)

Shootout = suite(
    name="shootout",
    benchmarks=[
        B("binarytrees", "binarytrees", 9),
        B("fannkuchredux", "fannkuch", 9),
        B("fasta", "fasta", 60000),
        B("fastaredux", "fastaredux", 80000),
        B("knucleotide", "knucleotide", 2000),
        B("mandelbrot_ascii", "mandelbrot", 300),
        B("mandelbrot_naive_ascii", "mandelbrot", 200),
        B("nbody", "nbody", 25000),
        B("nbody_naive", "nbody", 20000),
        B("pidigits", "pidigits", 30),
        B("regexdna", "regexdna", 500000),
        B("reversecomplement", "reversecomplement", 150000),
        B("spectralnorm", "spectralnorm", 1200),
        B("spectralnorm_math", "spectralnorm", 1200),
    ],
    parser=RebenchParser(),
    working_directory=lambda parameters, benchmark: (
        BENCHMARKS / "shootout" / benchmark.data[0]
    ),
    command=lambda parameters, benchmark: [
        str(Path(parameters.Rpath) / "bin" / "Rscript"),
        "harness.r",
        benchmark.name,
        str(parameters.iterations),
        str(benchmark.data[1]),
    ],
)


RealThing = suite(
    name="RealThing",
    working_directory=BENCHMARKS / "RealThing",
    benchmarks=[
        B("convolution", 500),
        B("convolution_slow", 1500),
        B("volcano", 1),
        B("flexclust", 5),
    ],
    parser=RebenchParser(),
    command=lambda parameters, benchmark: [
        str(Path(parameters.Rpath) / "bin" / "Rscript"),
        "harness.r",
        benchmark.name,
        str(parameters.iterations),
        str(benchmark.data),
    ],
)


Kaggles = suite(
    name="kaggle",
    benchmarks=[
        "basic-analysis",
        "bolt-driver",
        "london-airbnb",
        "placement",
        "titanic",
    ],
    parser=RebenchParser(),
    working_directory=lambda parameters, benchmark: INPUTS / "kaggle" / benchmark.name,
    command=lambda parameters, benchmark: [
        str(Path(parameters.Rpath) / "bin" / "Rscript"),
        "../../harness.r",
        benchmark.name,
        str(parameters.iterations),
    ],
)


Recommenderlab = suite(
    name="recommenderlab",
    benchmarks=["recommenderlab"],
    working_directory=INPUTS / "recommenderlab",
    parser=RebenchParser(),
    command=lambda parameters, benchmark: [
        str(Path(parameters.Rpath) / "bin" / "Rscript"),
        "runner.r",
    ],
)


conf = Config(
    [
        AreWeFast,
        Shootout,
        RealThing,
        Kaggles,
        Recommenderlab,
    ]
).env(LOCALE)

if __name__ == "__main__":
    main(conf, ["Rpath"], {"iterations": 15})
