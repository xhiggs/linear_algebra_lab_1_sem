import argparse
import importlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Поиск документов через ваш TF-IDF")
    parser.add_argument("query", nargs="?", default="matrix vector")
    args = parser.parse_args()
    impl = importlib.import_module("solution")
    path = Path(__file__).resolve().parent / "data/documents.json"
    documents = json.loads(path.read_text(encoding="utf-8"))
    try:
        vocabulary, idf, matrix = impl.tfidf_fit(documents)
        results = impl.rank_documents(args.query, matrix, vocabulary, idf)
    except NotImplementedError:
        parser.exit(1, "Заполните функции TF-IDF в solution.py.\n")
    for index, score in results:
        print(f"{index}: {score:.4f} | {documents[index]}")


if __name__ == "__main__":
    main()
