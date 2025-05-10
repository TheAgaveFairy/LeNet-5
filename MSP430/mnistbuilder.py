#!/usr/bin/env python3

import csv
import sys


def readFile(filename='mnist_test-1.csv'):
    rows = []
    labels = []
    with open(filename, 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            labels.append(row[0])
            rows.append(row[1:])
    return labels, rows


def formatRow(row):
    label = row[0]
    row = row[1:]
    temp = ", ".join(row)
    print(temp)
    return label, temp


def buildHeader(labels, rows, filename='mnist.h'):
    with open(filename, 'w') as file:
        file.write("#include <stdint.h>\n")
        file.write(f"#define MNISTSIZE {len(rows[0])}\n")
        file.write(f"#define NUMROWS {len(labels)}\n\n")

        file.write("#pragma PERSISTENT(labels)\n")
        file.write("const static uint8_t labels[] = {")
        for label in labels:
            file.write(f"{label}, ")
        file.write("};\n\n")

        file.write("#pragma PERSISTENT(mnist)\n")
        file.write("const static uint8_t mnist[][MNISTSIZE] = {\n")
        for row in rows:
            label, row = formatRow(row)

            file.write("\t{")
            file.write(str(row[:]))  # strip off [ ]
            file.write("},\n")
        file.write("};")


def main():
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        labels, rows = readFile(filename)
    else:
        labels, rows = readFile()

    NUM_ROWS = 10
    labels = labels[:NUM_ROWS]
    rows = rows[:NUM_ROWS]
    buildHeader(labels, rows)


if __name__ == "__main__":
    main()
