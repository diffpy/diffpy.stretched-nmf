import argparse

from diffpy.stretched_nmf.version import __version__  # noqa


def main():
    parser = argparse.ArgumentParser(
        prog="snmf",
        description=(
            "A python package implementing the stretched NMF algorithm.\n\n"
            "Currently, this project is used by importing "
            "`SNMFOptimizer` in Python rather than through a command-line "
            "workflow.\n\n"
            "For more information, visit: "
            "https://github.com/diffpy/diffpy.stretched-nmf/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action="store_true",
        help="Show the program's version number and exit",
    )

    args = parser.parse_args()

    if args.version:
        print(f"snmf {__version__}")
    else:
        # Default behavior when no arguments are given
        parser.print_help()


if __name__ == "__main__":
    main()
