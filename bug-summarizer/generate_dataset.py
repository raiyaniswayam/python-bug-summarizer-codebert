"""
generate_dataset.py — Synthetic Buggy Python Code Dataset Generator
=====================================================================
Generates ~4,500 labeled Python code snippets across 9 bug classes.

Key design decisions:
  - Each bug type has 15-20 "template families" (different code structures).
  - Within each family, variable names, function names, values are randomized.
  - Train/test split is done BY FAMILY INDEX to prevent data leakage.
  - Families 0-79% go to train, 80-89% to val, 90-100% to test.
  - Each sample has: code, bug_type, bug_description.
"""

import random
import csv
import os
import hashlib
from typing import List, Tuple, Dict

# ============================================================
# Random name pools (large, for variety)
# ============================================================
VARIABLE_NAMES = [
    "x", "y", "z", "a", "b", "n", "m", "val", "num", "count",
    "total", "result", "data", "item", "elem", "value", "temp",
    "acc", "idx", "pos", "key", "score", "price", "amount", "size",
    "length", "width", "height", "depth", "weight", "age", "limit",
    "threshold", "target", "current", "prev", "next_val", "sum_val",
    "product", "average", "minimum", "maximum", "delta", "offset",
    "capacity", "balance", "rate", "factor", "ratio", "margin",
]

FUNCTION_NAMES = [
    "calculate", "compute", "process", "analyze", "transform",
    "validate", "check", "find", "search", "filter_data",
    "sort_items", "merge", "combine", "aggregate", "summarize",
    "convert", "parse", "extract", "generate", "build",
    "evaluate", "measure", "estimate", "predict", "classify",
    "normalize", "scale", "adjust", "update", "refresh",
    "fetch", "load", "store", "save", "encode",
    "decode", "compress", "expand", "reduce", "accumulate",
    "get_total", "get_max", "get_min", "get_avg", "get_count",
    "run_task", "do_work", "handle", "resolve", "apply_rule",
]

LIST_NAMES = [
    "numbers", "values", "items", "elements", "data",
    "scores", "prices", "grades", "weights", "ages",
    "records", "entries", "samples", "points", "measurements",
    "results", "outputs", "inputs", "readings", "counts",
    "arr", "lst", "seq", "collection", "batch",
]

DICT_NAMES = [
    "config", "settings", "mapping", "lookup", "registry",
    "cache", "store", "table", "index", "catalog",
    "user_data", "metadata", "params", "options", "info",
]

STRING_VALS = [
    '"hello"', '"world"', '"test"', '"data"', '"result"',
    '"error"', '"success"', '"active"', '"pending"', '"done"',
    '"name"', '"value"', '"key"', '"item"', '"entry"',
]

# Seed for reproducibility
random.seed(42)


def pick(pool: list) -> str:
    """Pick a random item from a pool."""
    return random.choice(pool)


def pick_n(pool: list, n: int) -> list:
    """Pick n unique random items from a pool."""
    return random.sample(pool, min(n, len(pool)))


def rand_int(lo: int = 1, hi: int = 20) -> int:
    return random.randint(lo, hi)


def rand_list_literal(name: str = None) -> str:
    """Generate a random list literal like [3, 7, 1, 9, 2]."""
    length = random.randint(4, 8)
    elems = [str(random.randint(1, 100)) for _ in range(length)]
    return f"[{', '.join(elems)}]"


# ============================================================
# BUG TYPE 1: syntax_error
# ============================================================
def syntax_error_generators() -> List[callable]:
    """Return a list of generator functions for syntax_error."""
    generators = []

    # Family 1: Missing colon after if
    def family_1():
        v = pick(VARIABLE_NAMES)
        n = rand_int()
        code = f"def check({v}):\n    if {v} > {n}\n        print('big')\n    else:\n        print('small')"
        desc = f"Missing colon after 'if' statement on line 2"
        return code, desc
    generators.append(family_1)

    # Family 2: Missing colon after for
    def family_2():
        lst = pick(LIST_NAMES)
        v = pick(VARIABLE_NAMES)
        code = f"{lst} = {rand_list_literal()}\nfor {v} in {lst}\n    print({v})"
        desc = "Missing colon after 'for' loop header"
        return code, desc
    generators.append(family_2)

    # Family 3: Missing colon after while
    def family_3():
        v = pick(VARIABLE_NAMES)
        n = rand_int(5, 15)
        code = f"{v} = 0\nwhile {v} < {n}\n    print({v})\n    {v} += 1"
        desc = "Missing colon after 'while' loop header"
        return code, desc
    generators.append(family_3)

    # Family 4: Missing colon after def
    def family_4():
        fn = pick(FUNCTION_NAMES)
        v1, v2 = pick_n(VARIABLE_NAMES, 2)
        code = f"def {fn}({v1}, {v2})\n    return {v1} + {v2}"
        desc = "Missing colon after function definition"
        return code, desc
    generators.append(family_4)

    # Family 5: Unmatched parenthesis in function call
    def family_5():
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        n = rand_int()
        code = f"def {fn}({v}):\n    return {v} * 2\n\nresult = {fn}({n}"
        desc = "Unmatched opening parenthesis in function call"
        return code, desc
    generators.append(family_5)

    # Family 6: Unmatched bracket in list
    def family_6():
        lst = pick(LIST_NAMES)
        vals = ", ".join([str(rand_int()) for _ in range(4)])
        code = f"{lst} = [{vals}\nprint(len({lst}))"
        desc = "Unmatched opening bracket in list definition"
        return code, desc
    generators.append(family_6)

    # Family 7: Missing closing quote
    def family_7():
        v = pick(VARIABLE_NAMES)
        code = f'{v} = "hello world\nprint({v})'
        desc = "Missing closing quotation mark in string literal"
        return code, desc
    generators.append(family_7)

    # Family 8: Missing colon after elif
    def family_8():
        v = pick(VARIABLE_NAMES)
        n1, n2 = rand_int(1, 10), rand_int(11, 20)
        code = (f"def classify({v}):\n    if {v} < {n1}:\n        return 'low'\n"
                f"    elif {v} < {n2}\n        return 'medium'\n    else:\n        return 'high'")
        desc = "Missing colon after 'elif' statement"
        return code, desc
    generators.append(family_8)

    # Family 9: Missing colon after else
    def family_9():
        v = pick(VARIABLE_NAMES)
        n = rand_int()
        code = f"if {v} > {n}:\n    print('yes')\nelse\n    print('no')"
        desc = "Missing colon after 'else' statement"
        return code, desc
    generators.append(family_9)

    # Family 10: Missing colon after class
    def family_10():
        cls = random.choice(["Animal", "Vehicle", "Shape", "Student", "Product", "Account"])
        code = f"class {cls}\n    def __init__(self):\n        self.name = 'default'"
        desc = "Missing colon after class definition"
        return code, desc
    generators.append(family_10)

    # Family 11: Missing colon after try
    def family_11():
        v = pick(VARIABLE_NAMES)
        code = f"{v} = '123'\ntry\n    {v} = int({v})\nexcept ValueError:\n    print('invalid')"
        desc = "Missing colon after 'try' statement"
        return code, desc
    generators.append(family_11)

    # Family 12: Missing colon after except
    def family_12():
        v = pick(VARIABLE_NAMES)
        code = f"try:\n    {v} = int(input())\nexcept ValueError\n    print('bad input')"
        desc = "Missing colon after 'except' clause"
        return code, desc
    generators.append(family_12)

    # Family 13: Unmatched curly brace in dict
    def family_13():
        d = pick(DICT_NAMES)
        k1, k2 = pick_n(VARIABLE_NAMES, 2)
        code = f'{d} = {{"{k1}": 1, "{k2}": 2\nprint({d})'
        desc = "Unmatched opening curly brace in dictionary definition"
        return code, desc
    generators.append(family_13)

    # Family 14: Extra comma causing syntax error in print
    def family_14():
        v1, v2 = pick_n(VARIABLE_NAMES, 2)
        code = f"{v1} = {rand_int()}\n{v2} = {rand_int()}\nprint({v1},, {v2})"
        desc = "Extra comma causing syntax error in function call"
        return code, desc
    generators.append(family_14)

    # Family 15: Missing colon after with
    def family_15():
        fn = random.choice(["'data.txt'", "'output.csv'", "'log.txt'", "'config.json'"])
        v = random.choice(["f", "fp", "file", "handle"])
        code = f"with open({fn}, 'r') as {v}\n    content = {v}.read()\n    print(content)"
        desc = "Missing colon after 'with' statement"
        return code, desc
    generators.append(family_15)

    # Family 16: Missing closing parenthesis in condition
    def family_16():
        v = pick(VARIABLE_NAMES)
        fn = pick(FUNCTION_NAMES)
        code = f"def {fn}({v}):\n    if ({v} > 0 and {v} < 100:\n        return True\n    return False"
        desc = "Missing closing parenthesis in if-condition"
        return code, desc
    generators.append(family_16)

    return generators


# ============================================================
# BUG TYPE 2: off_by_one
# ============================================================
def off_by_one_generators() -> List[callable]:
    generators = []

    # Family 1: Accessing index len(lst) instead of len(lst)-1
    def family_1():
        lst = pick(LIST_NAMES)
        fn = pick(FUNCTION_NAMES)
        code = (f"def {fn}({lst}):\n    last = {lst}[len({lst})]\n    return last")
        desc = f"Off-by-one error: accessing index len({lst}) instead of len({lst})-1"
        return code, desc
    generators.append(family_1)

    # Family 2: range(1, n) when should be range(0, n) or range(n)
    def family_2():
        lst = pick(LIST_NAMES)
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}({lst}):\n    {v} = 0\n"
                f"    for i in range(1, len({lst})):\n        {v} += {lst}[i]\n    return {v}")
        desc = f"Off-by-one: range starts at 1 instead of 0, skipping the first element"
        return code, desc
    generators.append(family_2)

    # Family 3: range(n+1) for 0..n but should be range(n) for 0..n-1
    def family_3():
        lst = pick(LIST_NAMES)
        fn = pick(FUNCTION_NAMES)
        code = (f"def {fn}({lst}):\n    for i in range(len({lst}) + 1):\n"
                f"        print({lst}[i])")
        desc = f"Off-by-one: range goes one past the end, causing IndexError"
        return code, desc
    generators.append(family_3)

    # Family 4: Pairwise comparison missing last pair
    def family_4():
        lst = pick(LIST_NAMES)
        fn = pick(FUNCTION_NAMES)
        code = (f"def {fn}({lst}):\n    '''Check if list is sorted'''\n"
                f"    for i in range(len({lst})):\n"
                f"        if {lst}[i] > {lst}[i + 1]:\n            return False\n    return True")
        desc = f"Off-by-one: loop should use range(len({lst}) - 1) for pairwise comparison"
        return code, desc
    generators.append(family_4)

    # Family 5: Off-by-one in string slicing
    def family_5():
        v = pick(VARIABLE_NAMES)
        fn = pick(FUNCTION_NAMES)
        code = (f"def {fn}(text):\n    '''Get last n characters'''\n"
                f"    {v} = 3\n    return text[len(text) - {v} + 1:]")
        desc = "Off-by-one in string slicing: returns fewer characters than expected"
        return code, desc
    generators.append(family_5)

    # Family 6: Fence post error in counting
    def family_6():
        fn = pick(FUNCTION_NAMES)
        v1, v2 = pick_n(VARIABLE_NAMES, 2)
        code = (f"def {fn}({v1}, {v2}):\n    '''Count integers from {v1} to {v2} inclusive'''\n"
                f"    return {v2} - {v1}")
        desc = f"Fence-post error: should be {v2} - {v1} + 1 for inclusive count"
        return code, desc
    generators.append(family_6)

    # Family 7: While loop off by one
    def family_7():
        lst = pick(LIST_NAMES)
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}({lst}):\n    {v} = 0\n    i = 0\n"
                f"    while i <= len({lst}):\n        {v} += {lst}[i]\n        i += 1\n    return {v}")
        desc = f"Off-by-one: while loop condition should be < instead of <=, causes IndexError"
        return code, desc
    generators.append(family_7)

    # Family 8: Range excludes last element unintentionally
    def family_8():
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        n = rand_int(5, 15)
        code = (f"def {fn}({v}):\n    '''Sum from 1 to {v} inclusive'''\n"
                f"    total = 0\n    for i in range(1, {v}):\n        total += i\n    return total")
        desc = f"Off-by-one: range(1, {v}) excludes {v}, should be range(1, {v} + 1)"
        return code, desc
    generators.append(family_8)

    # Family 9: Wrong index in bubble sort inner loop
    def family_9():
        lst = pick(LIST_NAMES)
        fn = pick(FUNCTION_NAMES)
        code = (f"def {fn}({lst}):\n    n = len({lst})\n"
                f"    for i in range(n):\n        for j in range(n - 1):\n"
                f"            if {lst}[j] > {lst}[j + 1]:\n"
                f"                {lst}[j], {lst}[j + 1] = {lst}[j + 1], {lst}[j]\n"
                f"    return {lst}")
        # This is actually correct; let me make the bug: range(n) in inner loop
        code = (f"def {fn}({lst}):\n    n = len({lst})\n"
                f"    for i in range(n):\n        for j in range(n):\n"
                f"            if {lst}[j] > {lst}[j + 1]:\n"
                f"                {lst}[j], {lst}[j + 1] = {lst}[j + 1], {lst}[j]\n"
                f"    return {lst}")
        desc = f"Off-by-one: inner loop should be range(n - 1) to avoid IndexError on j+1"
        return code, desc
    generators.append(family_9)

    # Family 10: Reverse traversal off by one
    def family_10():
        lst = pick(LIST_NAMES)
        fn = pick(FUNCTION_NAMES)
        code = (f"def {fn}({lst}):\n    '''Print list in reverse'''\n"
                f"    for i in range(len({lst}), 0, -1):\n        print({lst}[i])")
        desc = f"Off-by-one: range starts at len({lst}) which is out of bounds, should start at len({lst})-1"
        return code, desc
    generators.append(family_10)

    # Family 11: Binary search off by one
    def family_11():
        lst = pick(LIST_NAMES)
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}({lst}, {v}):\n    lo, hi = 0, len({lst})\n"
                f"    while lo < hi:\n        mid = (lo + hi) // 2\n"
                f"        if {lst}[mid] == {v}:\n            return mid\n"
                f"        elif {lst}[mid] < {v}:\n            lo = mid\n"
                f"        else:\n            hi = mid\n    return -1")
        desc = "Off-by-one in binary search: lo = mid should be lo = mid + 1, may cause infinite loop"
        return code, desc
    generators.append(family_11)

    # Family 12: Matrix traversal off by one
    def family_12():
        fn = pick(FUNCTION_NAMES)
        code = (f"def {fn}(matrix):\n    rows = len(matrix)\n    cols = len(matrix[0])\n"
                f"    total = 0\n    for i in range(rows):\n"
                f"        for j in range(1, cols):\n"
                f"            total += matrix[i][j]\n    return total")
        desc = "Off-by-one: inner loop starts at 1 instead of 0, skipping the first column"
        return code, desc
    generators.append(family_12)

    # Family 13: Sliding window off by one
    def family_13():
        lst = pick(LIST_NAMES)
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}({lst}, {v}):\n    '''Find max sum of window size {v}'''\n"
                f"    max_sum = sum({lst}[:{v}])\n    current = max_sum\n"
                f"    for i in range({v}, len({lst}) + 1):\n"
                f"        current = current + {lst}[i] - {lst}[i - {v}]\n"
                f"        max_sum = max(max_sum, current)\n    return max_sum")
        desc = f"Off-by-one: range goes to len({lst}) + 1 instead of len({lst}), causes IndexError"
        return code, desc
    generators.append(family_13)

    # Family 14: Finding second largest off by one
    def family_14():
        lst = pick(LIST_NAMES)
        fn = pick(FUNCTION_NAMES)
        code = (f"def {fn}({lst}):\n    sorted_lst = sorted({lst})\n"
                f"    return sorted_lst[len(sorted_lst)]")
        desc = f"Off-by-one: index len(sorted_lst) is out of bounds; should be len(sorted_lst)-1 or -2"
        return code, desc
    generators.append(family_14)

    # Family 15: Chunk splitting off by one
    def family_15():
        lst = pick(LIST_NAMES)
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}({lst}, {v}):\n    '''Split list into chunks of size {v}'''\n"
                f"    chunks = []\n    for i in range(0, len({lst}), {v}):\n"
                f"        chunks.append({lst}[i:i + {v} - 1])\n    return chunks")
        desc = f"Off-by-one in slicing: slice end should be i + {v}, not i + {v} - 1"
        return code, desc
    generators.append(family_15)

    return generators


# ============================================================
# BUG TYPE 3: wrong_operator
# ============================================================
def wrong_operator_generators() -> List[callable]:
    generators = []

    # Family 1: Plus instead of minus in calculation
    def family_1():
        fn = pick(FUNCTION_NAMES)
        v1, v2 = pick_n(VARIABLE_NAMES, 2)
        code = (f"def {fn}({v1}, {v2}):\n    '''Calculate the difference'''\n"
                f"    return {v1} + {v2}")
        desc = f"Wrong operator: uses '+' instead of '-' to calculate the difference"
        return code, desc
    generators.append(family_1)

    # Family 2: Multiply instead of divide
    def family_2():
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        lst = pick(LIST_NAMES)
        code = (f"def {fn}({lst}):\n    '''Calculate average'''\n"
                f"    {v} = sum({lst})\n    return {v} * len({lst})")
        desc = f"Wrong operator: uses '*' instead of '/' to calculate the average"
        return code, desc
    generators.append(family_2)

    # Family 3: 'and' instead of 'or'
    def family_3():
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}({v}):\n    '''Check if value is out of range 0-100'''\n"
                f"    if {v} < 0 and {v} > 100:\n        return True\n    return False")
        desc = "Wrong operator: uses 'and' instead of 'or'; condition is impossible to satisfy"
        return code, desc
    generators.append(family_3)

    # Family 4: 'or' instead of 'and'
    def family_4():
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}({v}):\n    '''Check if value is within range 0-100'''\n"
                f"    if {v} >= 0 or {v} <= 100:\n        return True\n    return False")
        desc = "Wrong operator: uses 'or' instead of 'and'; condition is always True"
        return code, desc
    generators.append(family_4)

    # Family 5: += instead of -=
    def family_5():
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}({v}):\n    '''Countdown from {v} to 0'''\n"
                f"    while {v} > 0:\n        print({v})\n        {v} += 1")
        desc = f"Wrong operator: uses '+=' instead of '-=' causing infinite loop in countdown"
        return code, desc
    generators.append(family_5)

    # Family 6: > instead of <
    def family_6():
        fn = pick(FUNCTION_NAMES)
        lst = pick(LIST_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}({lst}):\n    '''Find minimum value'''\n"
                f"    {v} = {lst}[0]\n    for item in {lst}:\n"
                f"        if item > {v}:\n            {v} = item\n    return {v}")
        desc = f"Wrong operator: uses '>' instead of '<' so it finds maximum instead of minimum"
        return code, desc
    generators.append(family_6)

    # Family 7: Integer division instead of float division
    def family_7():
        fn = pick(FUNCTION_NAMES)
        v1, v2 = pick_n(VARIABLE_NAMES, 2)
        code = (f"def {fn}({v1}, {v2}):\n    '''Calculate precise ratio'''\n"
                f"    return {v1} // {v2}")
        desc = "Wrong operator: uses '//' (integer division) instead of '/' for precise ratio"
        return code, desc
    generators.append(family_7)

    # Family 8: Subtraction instead of addition in accumulator
    def family_8():
        fn = pick(FUNCTION_NAMES)
        lst = pick(LIST_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}({lst}):\n    '''Calculate total'''\n"
                f"    {v} = 0\n    for item in {lst}:\n"
                f"        {v} -= item\n    return {v}")
        desc = "Wrong operator: uses '-=' instead of '+=' for accumulating total"
        return code, desc
    generators.append(family_8)

    # Family 9: Wrong modulo operator
    def family_9():
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}({v}):\n    '''Check if number is even'''\n"
                f"    return {v} % 2 == 1")
        desc = f"Wrong comparison value: checks % 2 == 1 (odd) instead of % 2 == 0 (even)"
        return code, desc
    generators.append(family_9)

    # Family 10: Bitwise instead of logical
    def family_10():
        fn = pick(FUNCTION_NAMES)
        v1, v2 = pick_n(VARIABLE_NAMES, 2)
        code = (f"def {fn}({v1}, {v2}):\n    '''Check both are positive'''\n"
                f"    if {v1} > 0 & {v2} > 0:\n        return True\n    return False")
        desc = "Wrong operator: uses bitwise '&' instead of logical 'and' (operator precedence issue)"
        return code, desc
    generators.append(family_10)

    # Family 11: Power instead of multiply
    def family_11():
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}({v}):\n    '''Double the value'''\n    return {v} ** 2")
        desc = "Wrong operator: uses '**' (power) instead of '*' (multiply) for doubling"
        return code, desc
    generators.append(family_11)

    # Family 12: Not equal instead of equal in search
    def family_12():
        fn = pick(FUNCTION_NAMES)
        lst = pick(LIST_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}({lst}, {v}):\n    '''Find index of value'''\n"
                f"    for i, item in enumerate({lst}):\n"
                f"        if item != {v}:\n            return i\n    return -1")
        desc = f"Wrong operator: uses '!=' instead of '==' for finding a matching value"
        return code, desc
    generators.append(family_12)

    # Family 13: Wrong assignment in swap
    def family_13():
        fn = pick(FUNCTION_NAMES)
        lst = pick(LIST_NAMES)
        code = (f"def {fn}({lst}, i, j):\n    '''Swap two elements'''\n"
                f"    {lst}[i] = {lst}[j]\n    {lst}[j] = {lst}[i]")
        desc = "Wrong logic: sequential assignment overwrites first value; should use simultaneous swap"
        return code, desc
    generators.append(family_13)

    # Family 14: String concatenation with wrong operator
    def family_14():
        fn = pick(FUNCTION_NAMES)
        v1, v2 = pick_n(VARIABLE_NAMES, 2)
        code = (f"def {fn}({v1}, {v2}):\n    '''Combine two strings with space'''\n"
                f"    return {v1} - {v2}")
        desc = "Wrong operator: uses '-' on strings instead of '+' for concatenation"
        return code, desc
    generators.append(family_14)

    # Family 15: Wrong floor/ceil
    def family_15():
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"import math\n\ndef {fn}({v}):\n    '''Round up to nearest integer'''\n"
                f"    return math.floor({v})")
        desc = "Wrong function: uses math.floor (round down) instead of math.ceil (round up)"
        return code, desc
    generators.append(family_15)

    return generators


# ============================================================
# BUG TYPE 4: undefined_variable
# ============================================================
def undefined_variable_generators() -> List[callable]:
    generators = []

    # Family 1: Typo in variable name
    def family_1():
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        typo = v + "s" if not v.endswith("s") else v[:-1]
        code = (f"def {fn}():\n    {v} = 42\n    print({typo})")
        desc = f"Undefined variable: '{typo}' is used but only '{v}' was defined (typo)"
        return code, desc
    generators.append(family_1)

    # Family 2: Variable used before assignment
    def family_2():
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}():\n    print({v})\n    {v} = 10")
        desc = f"Variable '{v}' used before assignment"
        return code, desc
    generators.append(family_2)

    # Family 3: Accumulator not initialized
    def family_3():
        fn = pick(FUNCTION_NAMES)
        lst = pick(LIST_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}({lst}):\n    for item in {lst}:\n"
                f"        {v} += item\n    return {v}")
        desc = f"Undefined variable: '{v}' used as accumulator but never initialized"
        return code, desc
    generators.append(family_3)

    # Family 4: Counter not initialized
    def family_4():
        fn = pick(FUNCTION_NAMES)
        lst = pick(LIST_NAMES)
        v = pick(VARIABLE_NAMES)
        n = rand_int()
        code = (f"def {fn}({lst}):\n    for item in {lst}:\n"
                f"        if item > {n}:\n            {v} += 1\n    return {v}")
        desc = f"Undefined variable: counter '{v}' used but never initialized to 0"
        return code, desc
    generators.append(family_4)

    # Family 5: Wrong variable name in return
    def family_5():
        fn = pick(FUNCTION_NAMES)
        v1 = pick(VARIABLE_NAMES)
        v2 = pick(VARIABLE_NAMES)
        while v2 == v1:
            v2 = pick(VARIABLE_NAMES)
        code = (f"def {fn}(data):\n    {v1} = len(data)\n    return {v2}")
        desc = f"Undefined variable: returns '{v2}' but only '{v1}' was defined"
        return code, desc
    generators.append(family_5)

    # Family 6: Referencing variable from wrong scope
    def family_6():
        fn1, fn2 = pick_n(FUNCTION_NAMES, 2)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn1}():\n    {v} = 100\n\ndef {fn2}():\n    return {v} * 2")
        desc = f"Undefined variable: '{v}' defined in {fn1} but referenced in {fn2}"
        return code, desc
    generators.append(family_6)

    # Family 7: Missing import
    def family_7():
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}({v}):\n    return math.sqrt({v})")
        desc = "Undefined name: 'math' module used but never imported"
        return code, desc
    generators.append(family_7)

    # Family 8: Result variable never assigned in one branch
    def family_8():
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        r = "result"
        n = rand_int()
        code = (f"def {fn}({v}):\n    if {v} > {n}:\n        {r} = 'high'\n"
                f"    return {r}")
        desc = f"Undefined variable: '{r}' only defined inside if-block, undefined when condition is False"
        return code, desc
    generators.append(family_8)

    # Family 9: Misspelled function name
    def family_9():
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        typo_fn = fn[::-1] if len(fn) > 3 else fn + "_x"
        code = (f"def {fn}({v}):\n    return {v} ** 2\n\nresult = {typo_fn}(5)")
        desc = f"Undefined name: '{typo_fn}' called but the function is named '{fn}'"
        return code, desc
    generators.append(family_9)

    # Family 10: Using loop variable after loop (conceptual error)
    def family_10():
        fn = pick(FUNCTION_NAMES)
        lst = pick(LIST_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}():\n    {lst} = []\n    for {v} in {lst}:\n"
                f"        pass\n    print({v})")
        desc = f"Potential undefined variable: '{v}' used after loop, undefined if {lst} is empty"
        return code, desc
    generators.append(family_10)

    # Family 11: Variable defined in try but used outside
    def family_11():
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}():\n    try:\n        {v} = int('abc')\n"
                f"    except ValueError:\n        print('error')\n    return {v}")
        desc = f"Undefined variable: '{v}' only defined in try block, undefined if exception occurs"
        return code, desc
    generators.append(family_11)

    # Family 12: Using class attribute without self
    def family_12():
        cls = random.choice(["Calculator", "Counter", "Processor", "Manager"])
        v = pick(VARIABLE_NAMES)
        code = (f"class {cls}:\n    def __init__(self):\n        self.{v} = 0\n\n"
                f"    def increment(self):\n        {v} += 1")
        desc = f"Undefined variable: '{v}' should be 'self.{v}' to access instance attribute"
        return code, desc
    generators.append(family_12)

    # Family 13: Comprehension variable leak assumption
    def family_13():
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}():\n    doubled = [{v} * 2 for {v} in range(5)]\n"
                f"    tripled = {v} * 3\n    return tripled")
        desc = f"Fragile code: relies on '{v}' leaking from list comprehension scope"
        return code, desc
    generators.append(family_13)

    # Family 14: Global variable reference without global keyword
    def family_14():
        v = pick(VARIABLE_NAMES)
        fn = pick(FUNCTION_NAMES)
        code = (f"{v} = 0\n\ndef {fn}():\n    {v} += 1\n    return {v}")
        desc = f"UnboundLocalError: '{v}' referenced before assignment (needs 'global {v}')"
        return code, desc
    generators.append(family_14)

    return generators


# ============================================================
# BUG TYPE 5: null_reference
# ============================================================
def null_reference_generators() -> List[callable]:
    generators = []

    # Family 1: Calling method on None return
    def family_1():
        fn1, fn2 = pick_n(FUNCTION_NAMES, 2)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn1}({v}):\n    if {v} > 0:\n        return {v} * 2\n\n"
                f"def {fn2}():\n    result = {fn1}(-5)\n    print(result.bit_length())")
        desc = f"Null reference: {fn1} returns None for negative input, then .bit_length() is called on None"
        return code, desc
    generators.append(family_1)

    # Family 2: Indexing None
    def family_2():
        fn = pick(FUNCTION_NAMES)
        lst = pick(LIST_NAMES)
        code = (f"def {fn}({lst}):\n    result = {lst}.sort()\n    return result[0]")
        desc = f"Null reference: list.sort() returns None (sorts in-place), then None is indexed"
        return code, desc
    generators.append(family_2)

    # Family 3: Iterating over None
    def family_3():
        fn = pick(FUNCTION_NAMES)
        lst = pick(LIST_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}({lst}):\n    filtered = {lst}.sort()\n"
                f"    for {v} in filtered:\n        print({v})")
        desc = f"Null reference: iterating over None because sort() returns None"
        return code, desc
    generators.append(family_3)

    # Family 4: Arithmetic on None
    def family_4():
        fn = pick(FUNCTION_NAMES)
        d = pick(DICT_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}({d}):\n    {v} = {d}.get('price')\n    tax = {v} * 0.1\n    return tax")
        desc = f"Null reference: dict.get() returns None if key missing, then None is multiplied"
        return code, desc
    generators.append(family_4)

    # Family 5: Calling method on None from list.append
    def family_5():
        fn = pick(FUNCTION_NAMES)
        lst = pick(LIST_NAMES)
        code = (f"def {fn}():\n    {lst} = [1, 2, 3]\n"
                f"    {lst} = {lst}.append(4)\n    print(len({lst}))")
        desc = f"Null reference: list.append() returns None; reassigning it makes {lst} None"
        return code, desc
    generators.append(family_5)

    # Family 6: String concatenation with None
    def family_6():
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}({v}):\n    name = None\n    if {v} > 0:\n        name = 'positive'\n"
                f"    message = 'Result: ' + name\n    return message")
        desc = "Null reference: 'name' is None when condition is False, then string concatenation fails"
        return code, desc
    generators.append(family_6)

    # Family 7: Length of None
    def family_7():
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}():\n    {v} = None\n    if len({v}) > 0:\n        print({v})")
        desc = f"Null reference: calling len() on None value '{v}'"
        return code, desc
    generators.append(family_7)

    # Family 8: Attribute access on None
    def family_8():
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}({v}):\n    result = None\n    if {v}:\n        result = type({v})\n"
                f"    return result.__name__")
        desc = "Null reference: accessing .__name__ on None when condition is False"
        return code, desc
    generators.append(family_8)

    # Family 9: None from dict pop
    def family_9():
        fn = pick(FUNCTION_NAMES)
        d = pick(DICT_NAMES)
        code = (f"def {fn}({d}):\n    {d} = {{'a': 1, 'b': 2}}\n"
                f"    val = {d}.pop('c')\n    return val + 10")
        desc = "Null reference: dict.pop() with missing key raises KeyError (or returns None with default)"
        return code, desc
    generators.append(family_9)

    # Family 10: Chained None method call
    def family_10():
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}({v}):\n    result = {v}.strip().split(',')\n"
                f"    return result\n\noutput = {fn}(None)")
        desc = f"Null reference: calling .strip() on None passed as argument"
        return code, desc
    generators.append(family_10)

    # Family 11: None comparison with operations
    def family_11():
        fn = pick(FUNCTION_NAMES)
        lst = pick(LIST_NAMES)
        code = (f"def {fn}({lst}):\n    biggest = None\n    for item in {lst}:\n"
                f"        if item > biggest:\n            biggest = item\n    return biggest")
        desc = "Null reference: comparing item > None raises TypeError in Python 3"
        return code, desc
    generators.append(family_11)

    # Family 12: Using result of print()
    def family_12():
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}({v}):\n    output = print({v})\n    return output.upper()")
        desc = "Null reference: print() returns None, then .upper() is called on None"
        return code, desc
    generators.append(family_12)

    # Family 13: None from re.search when no match
    def family_13():
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"import re\n\ndef {fn}({v}):\n    match = re.search(r'\\d+', {v})\n"
                f"    return match.group()")
        desc = "Null reference: re.search returns None when no match, then .group() fails"
        return code, desc
    generators.append(family_13)

    # Family 14: Function with implicit None return used
    def family_14():
        fn1, fn2 = pick_n(FUNCTION_NAMES, 2)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn1}({v}):\n    print({v})\n\n"
                f"def {fn2}():\n    result = {fn1}(42)\n    return result + 1")
        desc = f"Null reference: {fn1}() has no return statement (returns None), then arithmetic on None"
        return code, desc
    generators.append(family_14)

    return generators


# ============================================================
# BUG TYPE 6: missing_return
# ============================================================
def missing_return_generators() -> List[callable]:
    generators = []

    # Family 1: Function computes sum but doesn't return
    def family_1():
        fn = pick(FUNCTION_NAMES)
        lst = pick(LIST_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}({lst}):\n    '''Calculate total sum'''\n"
                f"    {v} = 0\n    for item in {lst}:\n        {v} += item")
        desc = f"Missing return: function computes sum in '{v}' but never returns it"
        return code, desc
    generators.append(family_1)

    # Family 2: Function computes max but doesn't return
    def family_2():
        fn = pick(FUNCTION_NAMES)
        lst = pick(LIST_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}({lst}):\n    '''Find maximum value'''\n"
                f"    {v} = {lst}[0]\n    for item in {lst}:\n"
                f"        if item > {v}:\n            {v} = item")
        desc = f"Missing return: function finds maximum in '{v}' but never returns it"
        return code, desc
    generators.append(family_2)

    # Family 3: Function builds list but doesn't return
    def family_3():
        fn = pick(FUNCTION_NAMES)
        lst = pick(LIST_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}({lst}):\n    '''Filter positive values'''\n"
                f"    {v} = []\n    for item in {lst}:\n"
                f"        if item > 0:\n            {v}.append(item)")
        desc = f"Missing return: function builds filtered list '{v}' but never returns it"
        return code, desc
    generators.append(family_3)

    # Family 4: Boolean check function without return
    def family_4():
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        lst = pick(LIST_NAMES)
        code = (f"def {fn}({lst}):\n    '''Check if all values are positive'''\n"
                f"    {v} = True\n    for item in {lst}:\n"
                f"        if item <= 0:\n            {v} = False")
        desc = f"Missing return: function computes boolean '{v}' but never returns it"
        return code, desc
    generators.append(family_4)

    # Family 5: String builder without return
    def family_5():
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}(words):\n    '''Join words with commas'''\n"
                f"    {v} = ', '.join(words)")
        desc = f"Missing return: function creates joined string '{v}' but never returns it"
        return code, desc
    generators.append(family_5)

    # Family 6: Conditional return only in if, missing else
    def family_6():
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        n = rand_int()
        code = (f"def {fn}({v}):\n    '''Classify value'''\n"
                f"    if {v} > {n}:\n        return 'high'\n"
                f"    elif {v} > 0:\n        return 'medium'")
        desc = f"Missing return: function has no return for the case when {v} <= 0"
        return code, desc
    generators.append(family_6)

    # Family 7: Dictionary builder without return
    def family_7():
        fn = pick(FUNCTION_NAMES)
        lst = pick(LIST_NAMES)
        code = (f"def {fn}({lst}):\n    '''Count occurrences'''\n"
                f"    counts = {{}}\n    for item in {lst}:\n"
                f"        counts[item] = counts.get(item, 0) + 1")
        desc = "Missing return: function builds 'counts' dictionary but never returns it"
        return code, desc
    generators.append(family_7)

    # Family 8: Recursive function missing return on recursive call
    def family_8():
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}({v}):\n    '''Calculate factorial'''\n"
                f"    if {v} <= 1:\n        return 1\n    {fn}({v} - 1) * {v}")
        desc = f"Missing return: recursive call result is computed but not returned"
        return code, desc
    generators.append(family_8)

    # Family 9: Average computation without return
    def family_9():
        fn = pick(FUNCTION_NAMES)
        lst = pick(LIST_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}({lst}):\n    '''Calculate average'''\n"
                f"    {v} = sum({lst}) / len({lst})")
        desc = f"Missing return: function calculates average in '{v}' but never returns it"
        return code, desc
    generators.append(family_9)

    # Family 10: Sort function without return
    def family_10():
        fn = pick(FUNCTION_NAMES)
        lst = pick(LIST_NAMES)
        code = (f"def {fn}({lst}):\n    '''Sort in descending order'''\n"
                f"    sorted_list = sorted({lst}, reverse=True)")
        desc = "Missing return: function creates 'sorted_list' but never returns it"
        return code, desc
    generators.append(family_10)

    # Family 11: Min-max normalization without return
    def family_11():
        fn = pick(FUNCTION_NAMES)
        lst = pick(LIST_NAMES)
        code = (f"def {fn}({lst}):\n    '''Normalize values to 0-1 range'''\n"
                f"    min_val = min({lst})\n    max_val = max({lst})\n"
                f"    normalized = [(x - min_val) / (max_val - min_val) for x in {lst}]")
        desc = "Missing return: function computes 'normalized' list but never returns it"
        return code, desc
    generators.append(family_11)

    # Family 12: Matrix transpose without return
    def family_12():
        fn = pick(FUNCTION_NAMES)
        code = (f"def {fn}(matrix):\n    '''Transpose a matrix'''\n"
                f"    rows = len(matrix)\n    cols = len(matrix[0])\n"
                f"    transposed = [[matrix[j][i] for j in range(rows)] for i in range(cols)]")
        desc = "Missing return: function computes 'transposed' matrix but never returns it"
        return code, desc
    generators.append(family_12)

    # Family 13: String reversal without return
    def family_13():
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}({v}):\n    '''Reverse a string'''\n"
                f"    reversed_str = {v}[::-1]")
        desc = "Missing return: function creates 'reversed_str' but never returns it"
        return code, desc
    generators.append(family_13)

    # Family 14: Unique elements without return
    def family_14():
        fn = pick(FUNCTION_NAMES)
        lst = pick(LIST_NAMES)
        code = (f"def {fn}({lst}):\n    '''Get unique elements preserving order'''\n"
                f"    seen = set()\n    unique = []\n    for item in {lst}:\n"
                f"        if item not in seen:\n            seen.add(item)\n"
                f"            unique.append(item)")
        desc = "Missing return: function builds 'unique' list but never returns it"
        return code, desc
    generators.append(family_14)

    return generators


# ============================================================
# BUG TYPE 7: wrong_comparison
# ============================================================
def wrong_comparison_generators() -> List[callable]:
    generators = []

    # Family 1: >= instead of >
    def family_1():
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        lst = pick(LIST_NAMES)
        n = rand_int()
        code = (f"def {fn}({lst}):\n    '''Count values strictly greater than {n}'''\n"
                f"    {v} = 0\n    for item in {lst}:\n"
                f"        if item >= {n}:\n            {v} += 1\n    return {v}")
        desc = f"Wrong comparison: uses >= instead of > for 'strictly greater than {n}'"
        return code, desc
    generators.append(family_1)

    # Family 2: < instead of <=
    def family_2():
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        n = rand_int()
        code = (f"def {fn}({v}):\n    '''Check if value is at most {n}'''\n"
                f"    return {v} < {n}")
        desc = f"Wrong comparison: uses < instead of <= for 'at most {n}'"
        return code, desc
    generators.append(family_2)

    # Family 3: == 0 instead of != 0
    def family_3():
        fn = pick(FUNCTION_NAMES)
        lst = pick(LIST_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}({lst}):\n    '''Remove zeros from list'''\n"
                f"    {v} = []\n    for item in {lst}:\n"
                f"        if item == 0:\n            {v}.append(item)\n    return {v}")
        desc = "Wrong comparison: keeps zeros instead of removing them (== 0 instead of != 0)"
        return code, desc
    generators.append(family_3)

    # Family 4: Flipped if/else
    def family_4():
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}({v}):\n    '''Return absolute value'''\n"
                f"    if {v} < 0:\n        return {v}\n    else:\n        return -{v}")
        desc = "Wrong comparison: returns negative for positive input and positive for negative (flipped logic)"
        return code, desc
    generators.append(family_4)

    # Family 5: 'is' instead of '=='
    def family_5():
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        n = rand_int(100, 500)
        code = (f"def {fn}({v}):\n    '''Check if value equals {n}'''\n"
                f"    if {v} is {n}:\n        return True\n    return False")
        desc = f"Wrong comparison: uses 'is' (identity) instead of '==' (equality) for integer comparison"
        return code, desc
    generators.append(family_5)

    # Family 6: Wrong variable in comparison
    def family_6():
        fn = pick(FUNCTION_NAMES)
        v1, v2 = pick_n(VARIABLE_NAMES, 2)
        lst = pick(LIST_NAMES)
        code = (f"def {fn}({lst}, {v1}):\n    '''Check if {v1} is in list'''\n"
                f"    for {v2} in {lst}:\n"
                f"        if {v1} == {v1}:\n            return True\n    return False")
        desc = f"Wrong comparison: compares '{v1}' with itself instead of '{v2}'; always True"
        return code, desc
    generators.append(family_6)

    # Family 7: Negation error
    def family_7():
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}({v}):\n    '''Check if string is not empty'''\n"
                f"    if not len({v}) > 0:\n        return True\n    return False")
        desc = "Wrong comparison: 'not len > 0' returns True when string IS empty (inverted logic)"
        return code, desc
    generators.append(family_7)

    # Family 8: Wrong threshold direction
    def family_8():
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        lst = pick(LIST_NAMES)
        code = (f"def {fn}({lst}, {v}):\n    '''Filter values above threshold'''\n"
                f"    return [x for x in {lst} if x < {v}]")
        desc = f"Wrong comparison: filters values below threshold instead of above"
        return code, desc
    generators.append(family_8)

    # Family 9: Checking wrong end of range
    def family_9():
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        lo, hi = rand_int(1, 10), rand_int(50, 100)
        code = (f"def {fn}({v}):\n    '''Check if {v} is in range [{lo}, {hi}]'''\n"
                f"    return {v} >= {lo} and {v} >= {hi}")
        desc = f"Wrong comparison: second condition should be <= {hi}, not >= {hi}"
        return code, desc
    generators.append(family_9)

    # Family 10: Max finding with wrong comparison
    def family_10():
        fn = pick(FUNCTION_NAMES)
        lst = pick(LIST_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}({lst}):\n    '''Find second largest'''\n"
                f"    {lst}.sort()\n    if len({lst}) < 2:\n        return None\n"
                f"    return {lst}[1]")
        desc = "Wrong comparison: returns second element (second smallest) instead of second largest"
        return code, desc
    generators.append(family_10)

    # Family 11: Palindrome check wrong comparison
    def family_11():
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}({v}):\n    '''Check if string is a palindrome'''\n"
                f"    return {v} == {v}.sort() if isinstance({v}, list) else {v} == {v}[::1]")
        desc = "Wrong comparison: uses [::1] (forward) instead of [::-1] (reverse) for palindrome check"
        return code, desc
    generators.append(family_11)

    # Family 12: Age validation wrong boundary
    def family_12():
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}({v}):\n    '''Check if person is a teenager (13-19)'''\n"
                f"    return {v} > 13 and {v} < 19")
        desc = "Wrong comparison: excludes ages 13 and 19 (should use >= and <=)"
        return code, desc
    generators.append(family_12)

    # Family 13: Leap year wrong logic
    def family_13():
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}({v}):\n    '''Check if year is a leap year'''\n"
                f"    return {v} % 4 == 0 and {v} % 100 == 0")
        desc = "Wrong comparison: logic is incorrect for leap year (century years need % 400 check)"
        return code, desc
    generators.append(family_13)

    # Family 14: Empty check wrong way
    def family_14():
        fn = pick(FUNCTION_NAMES)
        lst = pick(LIST_NAMES)
        code = (f"def {fn}({lst}):\n    '''Process list if not empty'''\n"
                f"    if {lst} == None:\n        return 'empty'\n"
                f"    return sum({lst})")
        desc = f"Wrong comparison: checks for None instead of empty list; empty list is not None"
        return code, desc
    generators.append(family_14)

    return generators


# ============================================================
# BUG TYPE 8: indentation_error
# ============================================================
def indentation_error_generators() -> List[callable]:
    generators = []

    # Family 1: Return not indented inside function
    def family_1():
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}({v}):\n    result = {v} * 2\nreturn result")
        desc = "Indentation error: 'return' statement is outside the function body"
        return code, desc
    generators.append(family_1)

    # Family 2: Body not indented after if
    def family_2():
        v = pick(VARIABLE_NAMES)
        n = rand_int()
        code = (f"{v} = {n}\nif {v} > 0:\nprint('positive')")
        desc = "Indentation error: print statement should be indented inside the if block"
        return code, desc
    generators.append(family_2)

    # Family 3: Code inside loop that should be outside
    def family_3():
        fn = pick(FUNCTION_NAMES)
        lst = pick(LIST_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}({lst}):\n    {v} = 0\n    for item in {lst}:\n"
                f"        {v} += item\n        print(f'Total: {{{v}}}')")
        desc = "Indentation error: print is inside loop but should be after it (prints every iteration)"
        return code, desc
    generators.append(family_3)

    # Family 4: Code outside loop that should be inside
    def family_4():
        fn = pick(FUNCTION_NAMES)
        lst = pick(LIST_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}({lst}):\n    {v} = []\n    for item in {lst}:\n"
                f"        if item > 0:\n            pass\n"
                f"    {v}.append(item)")
        desc = f"Indentation error: append is outside the loop, only processes the last item"
        return code, desc
    generators.append(family_4)

    # Family 5: Else not aligned with if
    def family_5():
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        n = rand_int()
        code = (f"def {fn}({v}):\n    if {v} > {n}:\n        return 'high'\n"
                f"        else:\n            return 'low'")
        desc = "Indentation error: 'else' is incorrectly indented inside the if-block"
        return code, desc
    generators.append(family_5)

    # Family 6: Missing indentation after def
    def family_6():
        fn = pick(FUNCTION_NAMES)
        v1, v2 = pick_n(VARIABLE_NAMES, 2)
        code = (f"def {fn}({v1}, {v2}):\nresult = {v1} + {v2}\nreturn result")
        desc = "Indentation error: function body is not indented"
        return code, desc
    generators.append(family_6)

    # Family 7: Except not aligned with try
    def family_7():
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}({v}):\n    try:\n        return int({v})\n"
                f"        except ValueError:\n            return 0")
        desc = "Indentation error: 'except' is incorrectly indented inside the try-block"
        return code, desc
    generators.append(family_7)

    # Family 8: Inconsistent indentation (tabs vs spaces conceptual)
    def family_8():
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        n = rand_int()
        code = (f"def {fn}({v}):\n    if {v} > {n}:\n        result = 'big'\n"
                f"      return result")
        desc = "Indentation error: inconsistent indentation level (6 spaces instead of 4 or 8)"
        return code, desc
    generators.append(family_8)

    # Family 9: Nested block wrong level
    def family_9():
        fn = pick(FUNCTION_NAMES)
        lst = pick(LIST_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}({lst}):\n    for item in {lst}:\n"
                f"        if item > 0:\n        {v} = item * 2\n        print({v})")
        desc = "Indentation error: code inside if-block not properly indented (missing extra indent)"
        return code, desc
    generators.append(family_9)

    # Family 10: Class method wrong indentation
    def family_10():
        cls = random.choice(["Calculator", "Processor", "Handler", "Validator", "Converter"])
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"class {cls}:\n    def __init__(self):\n        self.{v} = 0\n\n"
                f"def {fn}(self):\n    return self.{v}")
        desc = f"Indentation error: method '{fn}' should be indented inside class '{cls}'"
        return code, desc
    generators.append(family_10)

    # Family 11: For body not indented
    def family_11():
        lst = pick(LIST_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"{lst} = {rand_list_literal()}\n{v} = 0\nfor item in {lst}:\n{v} += item\nprint({v})")
        desc = "Indentation error: loop body is not indented inside the for block"
        return code, desc
    generators.append(family_11)

    # Family 12: While body not indented
    def family_12():
        v = pick(VARIABLE_NAMES)
        n = rand_int(3, 10)
        code = (f"{v} = {n}\nwhile {v} > 0:\nprint({v})\n{v} -= 1")
        desc = "Indentation error: while loop body is not indented"
        return code, desc
    generators.append(family_12)

    # Family 13: Return inside loop instead of after
    def family_13():
        fn = pick(FUNCTION_NAMES)
        lst = pick(LIST_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}({lst}):\n    {v} = 0\n    for item in {lst}:\n"
                f"        {v} += item\n        return {v}")
        desc = f"Indentation error: return is inside the loop, causing it to return after first iteration"
        return code, desc
    generators.append(family_13)

    # Family 14: Break wrongly placed outside loop
    def family_14():
        fn = pick(FUNCTION_NAMES)
        lst = pick(LIST_NAMES)
        v = pick(VARIABLE_NAMES)
        n = rand_int()
        code = (f"def {fn}({lst}):\n    for {v} in {lst}:\n"
                f"        if {v} == {n}:\n            pass\n    break")
        desc = "Indentation error: 'break' is outside the for loop"
        return code, desc
    generators.append(family_14)

    return generators


# ============================================================
# BUG TYPE 9: no_bug (clean code)
# ============================================================
def no_bug_generators() -> List[callable]:
    generators = []

    # Family 1: Clean sum function
    def family_1():
        fn = pick(FUNCTION_NAMES)
        lst = pick(LIST_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}({lst}):\n    '''Calculate sum of all elements'''\n"
                f"    {v} = 0\n    for item in {lst}:\n        {v} += item\n    return {v}")
        desc = "No bug: correctly sums all elements in the list"
        return code, desc
    generators.append(family_1)

    # Family 2: Clean max function
    def family_2():
        fn = pick(FUNCTION_NAMES)
        lst = pick(LIST_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}({lst}):\n    '''Find the maximum value'''\n"
                f"    {v} = {lst}[0]\n    for item in {lst}[1:]:\n"
                f"        if item > {v}:\n            {v} = item\n    return {v}")
        desc = "No bug: correctly finds the maximum value in the list"
        return code, desc
    generators.append(family_2)

    # Family 3: Clean factorial
    def family_3():
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}({v}):\n    '''Calculate factorial recursively'''\n"
                f"    if {v} <= 1:\n        return 1\n    return {v} * {fn}({v} - 1)")
        desc = "No bug: correctly computes factorial using recursion"
        return code, desc
    generators.append(family_3)

    # Family 4: Clean binary search
    def family_4():
        fn = pick(FUNCTION_NAMES)
        lst = pick(LIST_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}({lst}, {v}):\n    '''Binary search in sorted list'''\n"
                f"    lo, hi = 0, len({lst}) - 1\n    while lo <= hi:\n"
                f"        mid = (lo + hi) // 2\n        if {lst}[mid] == {v}:\n"
                f"            return mid\n        elif {lst}[mid] < {v}:\n"
                f"            lo = mid + 1\n        else:\n            hi = mid - 1\n"
                f"    return -1")
        desc = "No bug: correctly implements binary search"
        return code, desc
    generators.append(family_4)

    # Family 5: Clean string reversal
    def family_5():
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}({v}):\n    '''Reverse a string'''\n"
                f"    return {v}[::-1]")
        desc = "No bug: correctly reverses a string using slicing"
        return code, desc
    generators.append(family_5)

    # Family 6: Clean dictionary counter
    def family_6():
        fn = pick(FUNCTION_NAMES)
        lst = pick(LIST_NAMES)
        code = (f"def {fn}({lst}):\n    '''Count occurrences of each element'''\n"
                f"    counts = {{}}\n    for item in {lst}:\n"
                f"        counts[item] = counts.get(item, 0) + 1\n    return counts")
        desc = "No bug: correctly counts occurrences using a dictionary"
        return code, desc
    generators.append(family_6)

    # Family 7: Clean list comprehension filter
    def family_7():
        fn = pick(FUNCTION_NAMES)
        lst = pick(LIST_NAMES)
        n = rand_int()
        code = (f"def {fn}({lst}):\n    '''Filter values greater than {n}'''\n"
                f"    return [x for x in {lst} if x > {n}]")
        desc = "No bug: correctly filters list using comprehension"
        return code, desc
    generators.append(family_7)

    # Family 8: Clean class with methods
    def family_8():
        cls = random.choice(["Stack", "Queue", "Counter", "Accumulator", "Buffer"])
        v = pick(VARIABLE_NAMES)
        code = (f"class {cls}:\n    def __init__(self):\n        self.{v} = []\n\n"
                f"    def push(self, item):\n        self.{v}.append(item)\n\n"
                f"    def pop(self):\n        if self.{v}:\n"
                f"            return self.{v}.pop()\n        return None\n\n"
                f"    def size(self):\n        return len(self.{v})")
        desc = f"No bug: correctly implements a {cls.lower()} data structure"
        return code, desc
    generators.append(family_8)

    # Family 9: Clean bubble sort
    def family_9():
        fn = pick(FUNCTION_NAMES)
        lst = pick(LIST_NAMES)
        code = (f"def {fn}({lst}):\n    '''Bubble sort implementation'''\n"
                f"    n = len({lst})\n    for i in range(n):\n"
                f"        for j in range(0, n - i - 1):\n"
                f"            if {lst}[j] > {lst}[j + 1]:\n"
                f"                {lst}[j], {lst}[j + 1] = {lst}[j + 1], {lst}[j]\n"
                f"    return {lst}")
        desc = "No bug: correctly implements bubble sort"
        return code, desc
    generators.append(family_9)

    # Family 10: Clean FizzBuzz
    def family_10():
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}({v}):\n    '''FizzBuzz for a single number'''\n"
                f"    if {v} % 15 == 0:\n        return 'FizzBuzz'\n"
                f"    elif {v} % 3 == 0:\n        return 'Fizz'\n"
                f"    elif {v} % 5 == 0:\n        return 'Buzz'\n"
                f"    return str({v})")
        desc = "No bug: correctly implements FizzBuzz logic"
        return code, desc
    generators.append(family_10)

    # Family 11: Clean two sum
    def family_11():
        fn = pick(FUNCTION_NAMES)
        lst = pick(LIST_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}({lst}, {v}):\n    '''Find two numbers that sum to target'''\n"
                f"    seen = {{}}\n    for i, num in enumerate({lst}):\n"
                f"        complement = {v} - num\n        if complement in seen:\n"
                f"            return [seen[complement], i]\n        seen[num] = i\n"
                f"    return []")
        desc = "No bug: correctly solves two-sum problem using hash map"
        return code, desc
    generators.append(family_11)

    # Family 12: Clean file reader with context manager
    def family_12():
        fn = pick(FUNCTION_NAMES)
        code = (f"def {fn}(filepath):\n    '''Read file contents safely'''\n"
                f"    try:\n        with open(filepath, 'r') as f:\n"
                f"            return f.read()\n    except FileNotFoundError:\n"
                f"        return None")
        desc = "No bug: correctly reads file with proper error handling"
        return code, desc
    generators.append(family_12)

    # Family 13: Clean GCD function
    def family_13():
        fn = pick(FUNCTION_NAMES)
        v1, v2 = pick_n(VARIABLE_NAMES, 2)
        code = (f"def {fn}({v1}, {v2}):\n    '''Calculate greatest common divisor'''\n"
                f"    while {v2}:\n        {v1}, {v2} = {v2}, {v1} % {v2}\n"
                f"    return {v1}")
        desc = "No bug: correctly computes GCD using Euclidean algorithm"
        return code, desc
    generators.append(family_13)

    # Family 14: Clean is_prime function
    def family_14():
        fn = pick(FUNCTION_NAMES)
        v = pick(VARIABLE_NAMES)
        code = (f"def {fn}({v}):\n    '''Check if a number is prime'''\n"
                f"    if {v} < 2:\n        return False\n"
                f"    for i in range(2, int({v} ** 0.5) + 1):\n"
                f"        if {v} % i == 0:\n            return False\n    return True")
        desc = "No bug: correctly checks if a number is prime"
        return code, desc
    generators.append(family_14)

    # Family 15: Clean flatten list
    def family_15():
        fn = pick(FUNCTION_NAMES)
        lst = pick(LIST_NAMES)
        code = (f"def {fn}({lst}):\n    '''Flatten a nested list'''\n"
                f"    result = []\n    for item in {lst}:\n"
                f"        if isinstance(item, list):\n"
                f"            result.extend({fn}(item))\n"
                f"        else:\n            result.append(item)\n    return result")
        desc = "No bug: correctly flattens nested lists using recursion"
        return code, desc
    generators.append(family_15)

    # Family 16: Clean matrix multiply
    def family_16():
        fn = pick(FUNCTION_NAMES)
        code = (f"def {fn}(a, b):\n    '''Multiply two matrices'''\n"
                f"    rows_a, cols_a = len(a), len(a[0])\n"
                f"    cols_b = len(b[0])\n"
                f"    result = [[0] * cols_b for _ in range(rows_a)]\n"
                f"    for i in range(rows_a):\n"
                f"        for j in range(cols_b):\n"
                f"            for k in range(cols_a):\n"
                f"                result[i][j] += a[i][k] * b[k][j]\n"
                f"    return result")
        desc = "No bug: correctly multiplies two matrices"
        return code, desc
    generators.append(family_16)

    # Family 17: Simple square
    def family_17():
        v = pick(VARIABLE_NAMES)
        return f"def square({v}):\n    return {v} * {v}", "No bug: simple square"
    generators.append(family_17)

    # Family 18: One-liner lambda
    def family_18():
        v = pick(VARIABLE_NAMES)
        return f"double = lambda {v}: {v} * 2", "No bug: lambda"
    generators.append(family_18)

    # Family 19: List comprehension
    def family_19():
        lst = pick(LIST_NAMES)
        return f"def get_evens({lst}):\n    return [x for x in {lst} if x % 2 == 0]", "No bug: list comp"
    generators.append(family_19)

    # Family 20: Simple math
    def family_20():
        return f"def add_two(a, b):\n    return a + b", "No bug: simple add"
    generators.append(family_20)

    # Family 21: Pass statement
    def family_21():
        fn = pick(FUNCTION_NAMES)
        return f"def {fn}():\n    pass", "No bug: pass"
    generators.append(family_21)

    # Family 22: Empty class
    def family_22():
        cls = pick(["Model", "View", "Controller", "Service"])
        return f"class {cls}:\n    pass", "No bug: empty class"
    generators.append(family_22)

    # Family 23: Direct return
    def family_23():
        return f"def get_constant():\n    return 42", "No bug: constant return"
    generators.append(family_23)

    return generators


# ============================================================
# MAIN: Generate and split the dataset
# ============================================================
BUG_TYPES = {
    "syntax_error": syntax_error_generators,
    "off_by_one": off_by_one_generators,
    "wrong_operator": wrong_operator_generators,
    "undefined_variable": undefined_variable_generators,
    "null_reference": null_reference_generators,
    "missing_return": missing_return_generators,
    "wrong_comparison": wrong_comparison_generators,
    "indentation_error": indentation_error_generators,
    "no_bug": no_bug_generators,
}

SAMPLES_PER_CLASS = 500


def stylize(code):
    lines = code.split('\n')
    if random.random() < 0.3:
        lines.insert(0, pick(["import os", "import sys", "import math", "import random", "from typing import List, Dict"]))
        lines.insert(1, "")
    if random.random() < 0.3:
        for i, line in enumerate(lines):
            if line.startswith("def ") and "'''" not in code and '"""' not in code:
                lines.insert(i+1, "    '''Docstring placeholder.'''")
                break
    if random.random() < 0.3:
        dummy = "def _helper_func():\n    pass\n"
        if random.random() < 0.5:
            lines.insert(0, dummy)
        else:
            lines.append("\n" + dummy)
    if random.random() < 0.4:
        lines.insert(0, f"# File ID: {random.randint(1000,9999)}")
    if random.random() < 0.2:
        # Add random empty lines
        if len(lines) > 2:
            idx = random.randint(1, len(lines)-1)
            lines.insert(idx, "")
    return '\n'.join(lines)


def generate_dataset():
    """
    Generate the full dataset with anti-leakage split.
    
    Split strategy: for each bug type, 80% of families go to train,
    10% to val, 10% to test. This ensures no template family appears
    in multiple splits.
    """
    all_samples = {"train": [], "val": [], "test": []}

    for bug_type, gen_func in BUG_TYPES.items():
        families = gen_func()
        n_families = len(families)

        # Split families into train/val/test (by family index)
        n_train = max(1, int(n_families * 0.8))
        n_val = max(1, int(n_families * 0.1))
        # Remaining go to test
        n_test = n_families - n_train - n_val
        if n_test < 1:
            n_train -= 1
            n_test = 1

        family_indices = list(range(n_families))
        random.shuffle(family_indices)

        train_families = family_indices[:n_train]
        val_families = family_indices[n_train:n_train + n_val]
        test_families = family_indices[n_train + n_val:]

        # Calculate samples per family for each split
        samples_train = int(SAMPLES_PER_CLASS * 0.8)
        samples_val = int(SAMPLES_PER_CLASS * 0.1)
        samples_test = SAMPLES_PER_CLASS - samples_train - samples_val

        # Generate samples for each split
        for split_name, family_idxs, n_samples in [
            ("train", train_families, samples_train),
            ("val", val_families, samples_val),
            ("test", test_families, samples_test),
        ]:
            per_family = max(1, n_samples // len(family_idxs))
            remainder = n_samples - per_family * len(family_idxs)

            for i, fam_idx in enumerate(family_idxs):
                count = per_family + (1 if i < remainder else 0)
                for _ in range(count):
                    code, description = families[fam_idx]()
                    code = stylize(code)
                    all_samples[split_name].append({
                        "code": code,
                        "bug_type": bug_type,
                        "bug_description": description,
                        "split": split_name,
                    })

        print(f"  {bug_type}: {len(train_families)} train families, "
              f"{len(val_families)} val families, {len(test_families)} test families")

    # Shuffle each split
    for split in all_samples:
        random.shuffle(all_samples[split])

    # Combine all
    all_rows = all_samples["train"] + all_samples["val"] + all_samples["test"]

    return all_rows


def main():
    print("=" * 60)
    print("Generating Bug Dataset")
    print("=" * 60)

    # Set output path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, "bug_dataset.csv")

    rows = generate_dataset()

    # Write to CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["code", "bug_type", "bug_description", "split"])
        writer.writeheader()
        writer.writerows(rows)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Dataset saved to: {output_path}")
    print(f"Total samples: {len(rows)}")

    # Count by split and class
    from collections import Counter
    split_counts = Counter(r["split"] for r in rows)
    class_counts = Counter(r["bug_type"] for r in rows)

    print(f"\nSplit distribution:")
    for split, count in sorted(split_counts.items()):
        print(f"  {split}: {count}")

    print(f"\nClass distribution:")
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls}: {count}")

    # Show a few samples
    print(f"\n{'=' * 60}")
    print("Sample entries:")
    print("=" * 60)
    for i in range(3):
        row = rows[i]
        print(f"\n--- Sample {i+1} ({row['split']}) ---")
        print(f"Bug type: {row['bug_type']}")
        print(f"Description: {row['bug_description']}")
        print(f"Code:\n{row['code']}")


if __name__ == "__main__":
    main()
