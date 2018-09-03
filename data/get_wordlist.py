import argparse
from glob import iglob


def collect_words(language):
    all_words = set()
    for path in iglob(f"monolingual/{language}/{language.upper()}_*.txt"):
        for line in open(path, 'r'):
            line = line.strip()
            words = line.split("\t")[:2]
            for word in words:
                all_words.add(word)
    for line in open(f"monolingual/{language}/questions-words.txt", 'r'):
        if line.startswith(": "):
            continue
        line = line.strip()
        for word in line.split(" "):
            all_words.add(word)
    for line in open(f"monolingual/{language}/categories.tsv", 'r'):
        line = line.strip()
        word, cat = line.split('\t')
        word = word.replace('""', '"')
        if word != "NULL":
            all_words.add(word)
    for word in all_words:
        print(word)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--language", type=str, required=True, help="The language to create the worlist for")
    args = parser.parse_args()
    collect_words(args.language)
