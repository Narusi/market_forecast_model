from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mforecast", description="Market forecast CLI")
    parser.add_argument("--version", action="store_true", help="Print version")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.version:
        print("market-forecast 0.1.0")
        return
    parser.print_help()


if __name__ == "__main__":
    main()
