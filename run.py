"""Проверки: python run.py unittest determinant; список: python run.py --list."""
import argparse
import importlib
import importlib.util
from pathlib import Path
import unittest
import sys
from tests import diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=["all", "algebra", "text"], default="all")
    parser.add_argument("--list", action="store_true", help="Показать доступные этапы")
    parser.add_argument("--no-plots", action="store_true", help="Без графиков; численные результаты сохраняются")
    parser.add_argument("--output", type=Path, help="Папка диагностических результатов тестов")
    parser.add_argument("mode", nargs="?", choices=["unittest"])
    parser.add_argument("test_name", nargs="?", help="Например determinant, tfidf, search")
    args = parser.parse_args()
    groups = sorted((Path(__file__).resolve().parent / "tests").glob("[0-9][0-9]_unittest_*_input"))
    names = {folder.name.split("_unittest_", 1)[1][:-6]: folder for folder in groups}
    if args.list:
        for name, folder in names.items():
            print(f"{folder.name[:2]}  python run.py unittest {name}")
        return
    if args.mode and not args.test_name:
        parser.error("После unittest укажите имя этапа; список: --list")
    if args.test_name:
        if args.part != "all":
            parser.error("Выберите конкретный этап или --part, не оба сразу")
        if args.test_name not in names:
            parser.error(f"Неизвестный этап {args.test_name!r}; список: --list")
        groups = [names[args.test_name]]
    elif args.part != "all":
        bounds = {"algebra": (1, 3), "text": (4, 9)}
        lo, hi = bounds[args.part]
        groups = [folder for folder in groups if lo <= int(folder.name[:2]) <= hi]
    module = importlib.import_module("solution")
    diagnostics.PLOTS = not args.no_plots
    diagnostics.OUTPUT = args.output or (Path(__file__).resolve().parent / "outputs/tests" /
                                         "student")
    suite = unittest.TestSuite()
    for folder in groups:
        spec = importlib.util.spec_from_file_location(folder.name, folder / "test.py")
        tests = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tests)
        tests.impl = module
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(tests))
    result = unittest.TextTestRunner(verbosity=2, stream=sys.stdout).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
