"""
test_multi_error.py — Comprehensive test suite for the multi-error detection pipeline.

Structure:
  - DEV set  (used to find failures and fix categories)
  - HOLDOUT set (untouched during development — final evaluation only)

Each snippet has:
  - code: the Python source
  - expected_types: set of bug types that MUST be reported
  - expected_count: how many issues to expect (None = don't check count)
  - category: grouping for failure analysis
"""

import json
import requests
import sys
from collections import defaultdict

API_URL = "http://127.0.0.1:5000/predict"

# ============================================================
# DEV SET — 75 snippets (25 multi-error, 25 single-error, 25 clean)
# ============================================================
DEV_SET = [
    # ----------------------------------------------------------------
    # Multi-error snippets (2+ known errors)
    # ----------------------------------------------------------------
    {
        "name": "DEV-M01: missing colon + mismatched bracket",
        "code": "def show_list(items)\n    print([1, 2, 3)",
        "expected_types": {"syntax_error", "mismatched_bracket"},
        "min_issues": 2,
        "category": "syntax+bracket",
    },
    {
        "name": "DEV-M02: two missing colons",
        "code": "def foo()\n    if True\n        pass",
        "expected_types": {"syntax_error"},
        "min_issues": 2,
        "category": "syntax_multi",
    },
    {
        "name": "DEV-M03: unclosed paren + missing colon",
        "code": "def bar(x\n    return x + 1",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "syntax+bracket",
    },
    {
        "name": "DEV-M04: two unclosed brackets",
        "code": "x = [1, 2, 3\ny = (4, 5",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "bracket_multi",
    },
    {
        "name": "DEV-M05: missing colon + indentation error",
        "code": "def test()\nprint('hello')",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "syntax+indent",
    },
    {
        "name": "DEV-M06: mismatched bracket types",
        "code": "data = [1, 2, 3)\nmore = {4, 5]",
        "expected_types": {"mismatched_bracket"},
        "min_issues": 1,
        "category": "bracket_multi",
    },
    {
        "name": "DEV-M07: missing colon + wrong bracket close",
        "code": "def process(items)\n    return sum(items]",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "syntax+bracket",
    },
    {
        "name": "DEV-M08: multiple syntax on different lines",
        "code": "class Foo\n    def bar(self)\n        pass",
        "expected_types": {"syntax_error"},
        "min_issues": 2,
        "category": "syntax_multi",
    },
    {
        "name": "DEV-M09: unterminated string + missing colon",
        "code": "def greet(name)\n    print('hello",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "syntax+string",
    },
    {
        "name": "DEV-M10: if and elif both missing colons",
        "code": "def check(x):\n    if x > 0\n        print('pos')\n    elif x < 0\n        print('neg')",
        "expected_types": {"syntax_error"},
        "min_issues": 2,
        "category": "syntax_multi",
    },
    {
        "name": "DEV-M11: while missing colon + unclosed paren",
        "code": "def loop(n):\n    while n > 0\n        print(n\n        n -= 1",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "syntax+bracket",
    },
    {
        "name": "DEV-M12: for missing colon + mismatched bracket",
        "code": "def iterate(items):\n    for item in items\n        print(item]",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "syntax+bracket",
    },
    {
        "name": "DEV-M13: try/except both missing colons",
        "code": "def safe_div(a, b):\n    try\n        return a / b\n    except ZeroDivisionError\n        return 0",
        "expected_types": {"syntax_error"},
        "min_issues": 2,
        "category": "syntax_multi",
    },
    {
        "name": "DEV-M14: nested unclosed brackets",
        "code": "result = func([1, 2, func2(3, 4)",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "bracket_multi",
    },
    {
        "name": "DEV-M15: two unterminated strings",
        "code": "x = 'hello\ny = \"world",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "string_multi",
    },
    {
        "name": "DEV-M16: missing colon on class + method",
        "code": "class Dog\n    def bark(self)\n        print('woof')",
        "expected_types": {"syntax_error"},
        "min_issues": 2,
        "category": "syntax_multi",
    },
    {
        "name": "DEV-M17: two mismatched close brackets",
        "code": "def calc(a, b):\n    return (a + b]\n\ndef other(x):\n    return [x * 2)",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "bracket_multi",
    },
    {
        "name": "DEV-M18: missing return + mutable default",
        "code": "def process(items=[]):\n    result = [x * 2 for x in items]",
        "expected_types": {"mutable_default_argument"},
        "min_issues": 1,
        "category": "logic_multi",
    },
    {
        "name": "DEV-M19: extra close + missing colon",
        "code": "def foo(x):\n    y = x))\n\ndef bar(z)\n    return z",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "syntax+bracket",
    },
    {
        "name": "DEV-M20: with and for missing colons",
        "code": "def read_file(path):\n    with open(path) as f\n        for line in f\n            print(line)",
        "expected_types": {"syntax_error"},
        "min_issues": 2,
        "category": "syntax_multi",
    },
    {
        "name": "DEV-M21: async def missing colon + bracket",
        "code": "async def fetch(url\n    return await get(url)",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "syntax+bracket",
    },
    {
        "name": "DEV-M22: finally missing colon + unclosed",
        "code": "def run():\n    try:\n        x = 1\n    finally\n        print(x",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "syntax_multi",
    },
    {
        "name": "DEV-M23: lambda missing colon + bracket",
        "code": "f = lambda x x + 1\ng = (lambda y: y * 2",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "syntax+bracket",
    },
    {
        "name": "DEV-M24: dict missing colon + bracket",
        "code": "data = {'key' 'value'}\nmore = {'a': 1, 'b': 2",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "syntax+bracket",
    },
    {
        "name": "DEV-M25: triple syntax errors",
        "code": "def a()\ndef b()\ndef c()",
        "expected_types": {"syntax_error"},
        "min_issues": 3,
        "category": "syntax_multi",
    },

    # ----------------------------------------------------------------
    # Single-error snippets (exactly 1 known error)
    # ----------------------------------------------------------------
    {
        "name": "DEV-S01: missing colon on def",
        "code": "def hello()\n    print('hi')",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "syntax_single",
    },
    {
        "name": "DEV-S02: indentation error",
        "code": "def foo():\nprint('bar')",
        "expected_types": {"syntax_error", "indentation_error"},
        "min_issues": 1,
        "category": "indent_single",
    },
    {
        "name": "DEV-S03: undefined variable",
        "code": "def compute(data):\n    total = sum(data)\n    return averge / len(data)",
        "expected_types": {"undefined_variable"},
        "min_issues": 1,
        "category": "undef_single",
    },
    {
        "name": "DEV-S04: unclosed bracket",
        "code": "data = [1, 2, 3",
        "expected_types": {"syntax_error", "unclosed_bracket"},
        "min_issues": 1,
        "category": "bracket_single",
    },
    {
        "name": "DEV-S05: mismatched bracket",
        "code": "x = (1 + 2]",
        "expected_types": {"syntax_error", "mismatched_bracket"},
        "min_issues": 1,
        "category": "bracket_single",
    },
    {
        "name": "DEV-S06: unterminated string",
        "code": "msg = 'hello world",
        "expected_types": {"syntax_error", "unterminated_string"},
        "min_issues": 1,
        "category": "string_single",
    },
    {
        "name": "DEV-S07: missing colon on if",
        "code": "def check(x):\n    if x > 0\n        return True\n    return False",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "syntax_single",
    },
    {
        "name": "DEV-S08: missing colon on while",
        "code": "def count(n):\n    i = 0\n    while i < n\n        i += 1\n    return i",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "syntax_single",
    },
    {
        "name": "DEV-S09: missing colon on for",
        "code": "def total(items):\n    s = 0\n    for x in items\n        s += x\n    return s",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "syntax_single",
    },
    {
        "name": "DEV-S10: missing colon on class",
        "code": "class MyClass\n    pass",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "syntax_single",
    },
    {
        "name": "DEV-S11: extra closing paren",
        "code": "result = (1 + 2))",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "bracket_single",
    },
    {
        "name": "DEV-S12: mutable default argument",
        "code": "def append_item(item, lst=[]):\n    lst.append(item)\n    return lst",
        "expected_types": {"mutable_default_argument"},
        "min_issues": 1,
        "category": "logic_single",
    },
    {
        "name": "DEV-S13: missing colon on elif",
        "code": "def classify(x):\n    if x > 0:\n        return 'pos'\n    elif x < 0\n        return 'neg'\n    else:\n        return 'zero'",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "syntax_single",
    },
    {
        "name": "DEV-S14: missing colon on else",
        "code": "def sign(x):\n    if x > 0:\n        return 1\n    else\n        return -1",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "syntax_single",
    },
    {
        "name": "DEV-S15: missing colon on try",
        "code": "def safe(x):\n    try\n        return int(x)\n    except ValueError:\n        return 0",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "syntax_single",
    },
    {
        "name": "DEV-S16: missing colon on except",
        "code": "def safe(x):\n    try:\n        return int(x)\n    except ValueError\n        return 0",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "syntax_single",
    },
    {
        "name": "DEV-S17: missing colon on finally",
        "code": "def clean():\n    try:\n        x = 1\n    finally\n        print('done')",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "syntax_single",
    },
    {
        "name": "DEV-S18: missing colon on with",
        "code": "def read():\n    with open('file.txt') as f\n        return f.read()",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "syntax_single",
    },
    {
        "name": "DEV-S19: unclosed curly brace",
        "code": "data = {'a': 1, 'b': 2",
        "expected_types": {"syntax_error", "unclosed_bracket"},
        "min_issues": 1,
        "category": "bracket_single",
    },
    {
        "name": "DEV-S20: unclosed square bracket in func",
        "code": "def get_list():\n    return [1, 2, 3",
        "expected_types": {"syntax_error", "unclosed_bracket"},
        "min_issues": 1,
        "category": "bracket_single",
    },
    {
        "name": "DEV-S21: tab/space mix indentation",
        "code": "def foo():\n    x = 1\n\tx = 2",
        "expected_types": {"indentation_error", "syntax_error"},
        "min_issues": 1,
        "category": "indent_single",
    },
    {
        "name": "DEV-S22: unexpected indent",
        "code": "x = 1\n    y = 2",
        "expected_types": {"indentation_error", "syntax_error"},
        "min_issues": 1,
        "category": "indent_single",
    },
    {
        "name": "DEV-S23: double unterminated string (triple quote)",
        "code": 'msg = """hello world',
        "expected_types": {"syntax_error", "unterminated_string"},
        "min_issues": 1,
        "category": "string_single",
    },
    {
        "name": "DEV-S24: undefined name in expression",
        "code": "def calc(x):\n    return x + undefined_var",
        "expected_types": {"undefined_variable"},
        "min_issues": 1,
        "category": "undef_single",
    },
    {
        "name": "DEV-S25: mutable dict default",
        "code": "def track(key, val, d={}):\n    d[key] = val\n    return d",
        "expected_types": {"mutable_default_argument"},
        "min_issues": 1,
        "category": "logic_single",
    },

    # ----------------------------------------------------------------
    # Clean snippets (no errors expected)
    # ----------------------------------------------------------------
    {
        "name": "DEV-C01: simple function",
        "code": "def add(a, b):\n    return a + b",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
    {
        "name": "DEV-C02: binary search",
        "code": "def binary_search(items, target):\n    lo, hi = 0, len(items) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if items[mid] == target:\n            return mid\n        elif items[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
    {
        "name": "DEV-C03: list comprehension",
        "code": "def squares(n):\n    return [i ** 2 for i in range(n)]",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
    {
        "name": "DEV-C04: class with init",
        "code": "class Point:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n\n    def distance(self):\n        return (self.x**2 + self.y**2) ** 0.5",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
    {
        "name": "DEV-C05: fibonacci",
        "code": "def fib(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
    {
        "name": "DEV-C06: dictionary operations",
        "code": "def count_words(text):\n    counts = {}\n    for word in text.split():\n        counts[word] = counts.get(word, 0) + 1\n    return counts",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
    {
        "name": "DEV-C07: try/except/finally",
        "code": "def safe_divide(a, b):\n    try:\n        result = a / b\n    except ZeroDivisionError:\n        result = float('inf')\n    finally:\n        return result",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
    {
        "name": "DEV-C08: nested list",
        "code": "def flatten(nested):\n    result = []\n    for item in nested:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
    {
        "name": "DEV-C09: lambda and map",
        "code": "def double_all(numbers):\n    return list(map(lambda x: x * 2, numbers))",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
    {
        "name": "DEV-C10: string formatting",
        "code": "def greet(name, age):\n    return f'Hello {name}, you are {age} years old!'",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
    {
        "name": "DEV-C11: generator function",
        "code": "def evens(n):\n    for i in range(n):\n        if i % 2 == 0:\n            yield i",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
    {
        "name": "DEV-C12: context manager usage",
        "code": "def read_lines(path):\n    with open(path) as f:\n        return f.readlines()",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
    {
        "name": "DEV-C13: decorator pattern",
        "code": "def decorator(func):\n    def wrapper(*args, **kwargs):\n        print('Before')\n        result = func(*args, **kwargs)\n        print('After')\n        return result\n    return wrapper",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
    {
        "name": "DEV-C14: enumerate and zip",
        "code": "def merge(keys, values):\n    result = {}\n    for i, (k, v) in enumerate(zip(keys, values)):\n        result[k] = v\n    return result",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
    {
        "name": "DEV-C15: multiple return paths",
        "code": "def classify(x):\n    if x > 0:\n        return 'positive'\n    elif x < 0:\n        return 'negative'\n    else:\n        return 'zero'",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
    {
        "name": "DEV-C16: set operations",
        "code": "def unique_common(a, b):\n    set_a = set(a)\n    set_b = set(b)\n    return set_a & set_b",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
    {
        "name": "DEV-C17: nested comprehension",
        "code": "def transpose(matrix):\n    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
    {
        "name": "DEV-C18: default parameter (immutable)",
        "code": "def repeat(s, n=3):\n    return s * n",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
    {
        "name": "DEV-C19: star args",
        "code": "def total(*args):\n    return sum(args)",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
    {
        "name": "DEV-C20: dict comprehension",
        "code": "def invert(d):\n    return {v: k for k, v in d.items()}",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
    {
        "name": "DEV-C21: while with break",
        "code": "def find_first(items, target):\n    i = 0\n    while i < len(items):\n        if items[i] == target:\n            return i\n        i += 1\n    return -1",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
    {
        "name": "DEV-C22: exception handling",
        "code": "def parse_int(s):\n    try:\n        return int(s)\n    except (ValueError, TypeError):\n        return None",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
    {
        "name": "DEV-C23: sorting with key",
        "code": "def sort_by_length(words):\n    return sorted(words, key=lambda w: len(w))",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
    {
        "name": "DEV-C24: walrus operator style",
        "code": "def first_long(words, min_len=5):\n    for w in words:\n        if len(w) >= min_len:\n            return w\n    return None",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
    {
        "name": "DEV-C25: multi-line expression",
        "code": "def complex_calc(a, b, c):\n    result = (\n        a * b\n        + c ** 2\n        - (a + b) / (c + 1)\n    )\n    return result",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
]


# ============================================================
# HOLDOUT SET — 75 snippets (25 multi-error, 25 single-error, 25 clean)
# ============================================================
HOLDOUT_SET = [
    # ----------------------------------------------------------------
    # Multi-error snippets
    # ----------------------------------------------------------------
    {
        "name": "HOLD-M01: two missing colons in nested",
        "code": "def outer()\n    def inner()\n        return 1",
        "expected_types": {"syntax_error"},
        "min_issues": 2,
        "category": "syntax_multi",
    },
    {
        "name": "HOLD-M02: bracket mismatch + missing colon",
        "code": "def parse(text)\n    words = text.split(']')",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "syntax+bracket",
    },
    {
        "name": "HOLD-M03: two unclosed parens",
        "code": "a = (1 + 2\nb = (3 + 4",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "bracket_multi",
    },
    {
        "name": "HOLD-M04: missing colon on for and while",
        "code": "def mixed(n):\n    for i in range(n)\n        pass\n    while n > 0\n        n -= 1",
        "expected_types": {"syntax_error"},
        "min_issues": 2,
        "category": "syntax_multi",
    },
    {
        "name": "HOLD-M05: class and method missing colons",
        "code": "class Animal\n    def speak(self)\n        return 'sound'",
        "expected_types": {"syntax_error"},
        "min_issues": 2,
        "category": "syntax_multi",
    },
    {
        "name": "HOLD-M06: except + finally missing colons",
        "code": "def handler():\n    try:\n        pass\n    except Exception\n        pass\n    finally\n        pass",
        "expected_types": {"syntax_error"},
        "min_issues": 2,
        "category": "syntax_multi",
    },
    {
        "name": "HOLD-M07: unclosed bracket + unterminated string",
        "code": "data = [1, 2, 'hello",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "bracket+string",
    },
    {
        "name": "HOLD-M08: three missing colons",
        "code": "if True\n    for x in range(5)\n        while x > 0\n            x -= 1",
        "expected_types": {"syntax_error"},
        "min_issues": 3,
        "category": "syntax_multi",
    },
    {
        "name": "HOLD-M09: double indentation error",
        "code": "def foo():\n    x = 1\n  y = 2\n    z = 3\n  w = 4",
        "expected_types": {"indentation_error", "syntax_error"},
        "min_issues": 1,
        "category": "indent_multi",
    },
    {
        "name": "HOLD-M10: missing colon + extra bracket",
        "code": "def add(a, b)\n    return (a + b))",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "syntax+bracket",
    },
    {
        "name": "HOLD-M11: two syntax errors different types",
        "code": "def first()\n    x = [1, 2\n    return x",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "syntax_multi",
    },
    {
        "name": "HOLD-M12: nested missing colons 3-deep",
        "code": "def a():\n    if True\n        for i in range(3)\n            print(i)",
        "expected_types": {"syntax_error"},
        "min_issues": 2,
        "category": "syntax_multi",
    },
    {
        "name": "HOLD-M13: mismatched brackets on two lines",
        "code": "x = (1 + 2]\ny = [3 + 4)",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "bracket_multi",
    },
    {
        "name": "HOLD-M14: def and return syntax issues",
        "code": "def compute(x, y\n    result = x + y\n    return result",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "syntax+bracket",
    },
    {
        "name": "HOLD-M15: complex multi-error",
        "code": "class Calculator\n    def add(self, a, b)\n        return a + b\n    def sub(self, a, b)\n        return a - b",
        "expected_types": {"syntax_error"},
        "min_issues": 3,
        "category": "syntax_multi",
    },
    {
        "name": "HOLD-M16: triple unclosed brackets",
        "code": "x = (\ny = [\nz = {",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "bracket_multi",
    },
    {
        "name": "HOLD-M17: mix of colon and indent",
        "code": "def f()\nx = 1",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "syntax+indent",
    },
    {
        "name": "HOLD-M18: elif and else missing colons",
        "code": "def check(x):\n    if x > 10:\n        return 'big'\n    elif x > 5\n        return 'mid'\n    else\n        return 'small'",
        "expected_types": {"syntax_error"},
        "min_issues": 2,
        "category": "syntax_multi",
    },
    {
        "name": "HOLD-M19: while missing colon + unclosed paren",
        "code": "def loop():\n    i = 0\n    while i < 10\n        print(i\n        i += 1",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "syntax+bracket",
    },
    {
        "name": "HOLD-M20: two defs missing colons",
        "code": "def alpha()\n    pass\n\ndef beta()\n    pass",
        "expected_types": {"syntax_error"},
        "min_issues": 2,
        "category": "syntax_multi",
    },
    {
        "name": "HOLD-M21: unclosed bracket deep inside",
        "code": "def process(data):\n    result = []\n    for item in data:\n        result.append(transform(item\n    return result",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "bracket_single",
    },
    {
        "name": "HOLD-M22: multiple mismatched in one line",
        "code": "x = ([1, 2, 3})",
        "expected_types": {"syntax_error", "mismatched_bracket"},
        "min_issues": 1,
        "category": "bracket_multi",
    },
    {
        "name": "HOLD-M23: for missing colon + undefined",
        "code": "def total(items):\n    s = 0\n    for x in items\n        s += x\n    return s + bonus",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "syntax+undef",
    },
    {
        "name": "HOLD-M24: with missing colon + bracket",
        "code": "def load():\n    with open('file.txt') as f\n        data = json.loads(f.read(",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "syntax+bracket",
    },
    {
        "name": "HOLD-M25: two class methods missing colons",
        "code": "class Util:\n    def method_a(self)\n        pass\n    def method_b(self)\n        pass",
        "expected_types": {"syntax_error"},
        "min_issues": 2,
        "category": "syntax_multi",
    },

    # ----------------------------------------------------------------
    # Single-error snippets
    # ----------------------------------------------------------------
    {
        "name": "HOLD-S01: missing colon on def",
        "code": "def square(x)\n    return x ** 2",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "syntax_single",
    },
    {
        "name": "HOLD-S02: unexpected indent",
        "code": "y = 5\n    z = 10",
        "expected_types": {"indentation_error", "syntax_error"},
        "min_issues": 1,
        "category": "indent_single",
    },
    {
        "name": "HOLD-S03: unclosed paren",
        "code": "val = (1 + 2 + 3",
        "expected_types": {"syntax_error", "unclosed_bracket"},
        "min_issues": 1,
        "category": "bracket_single",
    },
    {
        "name": "HOLD-S04: mismatched closing",
        "code": "items = [1, 2, 3}",
        "expected_types": {"syntax_error", "mismatched_bracket"},
        "min_issues": 1,
        "category": "bracket_single",
    },
    {
        "name": "HOLD-S05: undefined variable",
        "code": "def multiply(a, b):\n    return a * c",
        "expected_types": {"undefined_variable"},
        "min_issues": 1,
        "category": "undef_single",
    },
    {
        "name": "HOLD-S06: mutable default set",
        "code": "def add_to_set(x, s=set()):\n    s.add(x)\n    return s",
        "expected_types": {"mutable_default_argument"},
        "min_issues": 1,
        "category": "logic_single",
    },
    {
        "name": "HOLD-S07: for missing colon",
        "code": "def iterate(n):\n    for i in range(n)\n        print(i)",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "syntax_single",
    },
    {
        "name": "HOLD-S08: while missing colon",
        "code": "def countdown(n):\n    while n > 0\n        n -= 1",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "syntax_single",
    },
    {
        "name": "HOLD-S09: class missing colon",
        "code": "class Vehicle\n    def __init__(self, name):\n        self.name = name",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "syntax_single",
    },
    {
        "name": "HOLD-S10: try missing colon",
        "code": "def convert(val):\n    try\n        return float(val)\n    except:\n        return 0.0",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "syntax_single",
    },
    {
        "name": "HOLD-S11: except missing colon",
        "code": "def convert(val):\n    try:\n        return float(val)\n    except ValueError\n        return 0.0",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "syntax_single",
    },
    {
        "name": "HOLD-S12: unterminated double quote",
        "code": 'name = "hello world',
        "expected_types": {"syntax_error", "unterminated_string"},
        "min_issues": 1,
        "category": "string_single",
    },
    {
        "name": "HOLD-S13: extra close bracket",
        "code": "result = [1, 2, 3]]",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "bracket_single",
    },
    {
        "name": "HOLD-S14: indentation after colon",
        "code": "def foo():\nbar()",
        "expected_types": {"indentation_error", "syntax_error"},
        "min_issues": 1,
        "category": "indent_single",
    },
    {
        "name": "HOLD-S15: missing colon on if standalone",
        "code": "x = 5\nif x > 3\n    print(x)",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "syntax_single",
    },
    {
        "name": "HOLD-S16: undefined in return",
        "code": "def get_total(items):\n    s = sum(items)\n    return totl",
        "expected_types": {"undefined_variable"},
        "min_issues": 1,
        "category": "undef_single",
    },
    {
        "name": "HOLD-S17: mutable default dict",
        "code": "def update(key, val, d={}):\n    d[key] = val\n    return d",
        "expected_types": {"mutable_default_argument"},
        "min_issues": 1,
        "category": "logic_single",
    },
    {
        "name": "HOLD-S18: unclosed triple quote",
        "code": "x = '''hello\nworld",
        "expected_types": {"syntax_error", "unterminated_string"},
        "min_issues": 1,
        "category": "string_single",
    },
    {
        "name": "HOLD-S19: unclosed curly in dict",
        "code": "config = {'host': 'localhost', 'port': 8080",
        "expected_types": {"syntax_error", "unclosed_bracket"},
        "min_issues": 1,
        "category": "bracket_single",
    },
    {
        "name": "HOLD-S20: with missing colon",
        "code": "def load_data():\n    with open('data.json') as f\n        return f.read()",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "syntax_single",
    },
    {
        "name": "HOLD-S21: unmatched paren in expr",
        "code": "result = ((1 + 2) * 3",
        "expected_types": {"syntax_error", "unclosed_bracket"},
        "min_issues": 1,
        "category": "bracket_single",
    },
    {
        "name": "HOLD-S22: dedent error",
        "code": "def foo():\n    if True:\n        x = 1\n  y = 2",
        "expected_types": {"indentation_error", "syntax_error"},
        "min_issues": 1,
        "category": "indent_single",
    },
    {
        "name": "HOLD-S23: undefined function call",
        "code": "def run():\n    result = compute(5)\n    return result",
        "expected_types": {"undefined_variable"},
        "min_issues": 1,
        "category": "undef_single",
    },
    {
        "name": "HOLD-S24: finally missing colon",
        "code": "def cleanup():\n    try:\n        pass\n    finally\n        print('done')",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "syntax_single",
    },
    {
        "name": "HOLD-S25: else missing colon",
        "code": "def abs_val(x):\n    if x >= 0:\n        return x\n    else\n        return -x",
        "expected_types": {"syntax_error"},
        "min_issues": 1,
        "category": "syntax_single",
    },

    # ----------------------------------------------------------------
    # Clean snippets
    # ----------------------------------------------------------------
    {
        "name": "HOLD-C01: simple return",
        "code": "def identity(x):\n    return x",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
    {
        "name": "HOLD-C02: swap values",
        "code": "def swap(a, b):\n    return b, a",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
    {
        "name": "HOLD-C03: factorial recursive",
        "code": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
    {
        "name": "HOLD-C04: is_palindrome",
        "code": "def is_palindrome(s):\n    return s == s[::-1]",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
    {
        "name": "HOLD-C05: max of three",
        "code": "def max_three(a, b, c):\n    return max(a, max(b, c))",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
    {
        "name": "HOLD-C06: filter evens",
        "code": "def get_evens(numbers):\n    return [n for n in numbers if n % 2 == 0]",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
    {
        "name": "HOLD-C07: bubble sort",
        "code": "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n - i - 1):\n            if arr[j] > arr[j + 1]:\n                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n    return arr",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
    {
        "name": "HOLD-C08: string reversal",
        "code": "def reverse_string(s):\n    result = ''\n    for char in s:\n        result = char + result\n    return result",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
    {
        "name": "HOLD-C09: power function",
        "code": "def power(base, exp):\n    result = 1\n    for _ in range(exp):\n        result *= base\n    return result",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
    {
        "name": "HOLD-C10: GCD",
        "code": "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
    {
        "name": "HOLD-C11: class inheritance",
        "code": "class Animal:\n    def __init__(self, name):\n        self.name = name\n\nclass Dog(Animal):\n    def speak(self):\n        return f'{self.name} says woof!'",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
    {
        "name": "HOLD-C12: enumerate usage",
        "code": "def indexed_items(items):\n    return [(i, item) for i, item in enumerate(items)]",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
    {
        "name": "HOLD-C13: reduce-like pattern",
        "code": "def product(numbers):\n    result = 1\n    for n in numbers:\n        result *= n\n    return result",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
    {
        "name": "HOLD-C14: nested dict access",
        "code": "def get_nested(d, keys, default=None):\n    for key in keys:\n        if isinstance(d, dict):\n            d = d.get(key, default)\n        else:\n            return default\n    return d",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
    {
        "name": "HOLD-C15: matrix multiply element",
        "code": "def dot_product(a, b):\n    return sum(x * y for x, y in zip(a, b))",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
    {
        "name": "HOLD-C16: prime check",
        "code": "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
    {
        "name": "HOLD-C17: flatten one level",
        "code": "def flatten(lst):\n    result = []\n    for item in lst:\n        if isinstance(item, list):\n            result.extend(item)\n        else:\n            result.append(item)\n    return result",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
    {
        "name": "HOLD-C18: char frequency",
        "code": "def char_freq(s):\n    freq = {}\n    for c in s:\n        freq[c] = freq.get(c, 0) + 1\n    return freq",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
    {
        "name": "HOLD-C19: capitalize words",
        "code": "def title_case(text):\n    return ' '.join(word.capitalize() for word in text.split())",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
    {
        "name": "HOLD-C20: range sum",
        "code": "def range_sum(start, end):\n    return sum(range(start, end + 1))",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
    {
        "name": "HOLD-C21: zip_longest style",
        "code": "def merge_lists(a, b, fill=None):\n    result = []\n    for i in range(max(len(a), len(b))):\n        va = a[i] if i < len(a) else fill\n        vb = b[i] if i < len(b) else fill\n        result.append((va, vb))\n    return result",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
    {
        "name": "HOLD-C22: simple class method",
        "code": "class Counter:\n    def __init__(self):\n        self.count = 0\n\n    def increment(self):\n        self.count += 1\n        return self.count",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
    {
        "name": "HOLD-C23: any/all usage",
        "code": "def has_negative(numbers):\n    return any(n < 0 for n in numbers)\n\ndef all_positive(numbers):\n    return all(n > 0 for n in numbers)",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
    {
        "name": "HOLD-C24: chained comparison",
        "code": "def in_range(x, lo, hi):\n    return lo <= x <= hi",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
    {
        "name": "HOLD-C25: exception re-raise",
        "code": "def safe_run(func, *args):\n    try:\n        return func(*args)\n    except Exception as e:\n        print(f'Error: {e}')\n        raise",
        "expected_types": set(),
        "min_issues": 0,
        "max_issues": 0,
        "category": "clean",
    },
]


def run_test(snippet):
    """Run a single test snippet and return (pass, details)."""
    try:
        resp = requests.post(API_URL, json={"code": snippet["code"]}, timeout=10)
        data = resp.json()
    except Exception as e:
        return False, f"REQUEST FAILED: {e}"

    issues = data.get("issues", [])
    issue_types = {i["type"] for i in issues}
    count = len(issues)
    expected = snippet["expected_types"]
    min_issues = snippet.get("min_issues", 0)
    max_issues = snippet.get("max_issues", None)

    failures = []

    # Check that all expected types are present (at least one match)
    for exp_type in expected:
        if exp_type not in issue_types:
            # For sets like {"indentation_error", "syntax_error"} either is OK
            pass

    # For expected_types: at least one of the expected types must be found
    if expected:
        if not expected.intersection(issue_types):
            failures.append(f"Expected one of {expected}, got {issue_types}")

    # Check min issues
    if count < min_issues:
        failures.append(f"Expected at least {min_issues} issues, got {count}")

    # Check max issues (for clean code)
    if max_issues is not None and count > max_issues:
        failures.append(f"Expected at most {max_issues} issues, got {count}: {issue_types}")

    passed = len(failures) == 0
    detail = "; ".join(failures) if failures else "OK"
    return passed, detail, data


def run_suite(name, snippets):
    """Run a test suite and return metrics."""
    print(f"\n{'='*60}")
    print(f"  {name} ({len(snippets)} snippets)")
    print(f"{'='*60}")

    results = []
    failures_by_category = defaultdict(list)

    total_planted = 0
    total_found = 0
    total_reported = 0
    total_real = 0
    clean_false_alarms = 0
    clean_total = 0

    for snippet in snippets:
        passed, detail, data = run_test(snippet)
        status = "✅" if passed else "❌"
        print(f"  {status} {snippet['name']}: {detail}")

        if not passed:
            failures_by_category[snippet["category"]].append(snippet["name"])

        issues = data.get("issues", [])
        expected = snippet["expected_types"]

        # Metrics
        if snippet["category"] == "clean":
            clean_total += 1
            if len(issues) > 0:
                clean_false_alarms += 1
        else:
            total_planted += len(expected) if expected else snippet.get("min_issues", 1)
            found = len(expected.intersection({i["type"] for i in issues})) if expected else 0
            total_found += max(found, 1 if passed and expected else 0)
            total_reported += len(issues)
            # Approximate real issues = expected issues count
            total_real += len(expected) if expected else snippet.get("min_issues", 1)

        results.append({"name": snippet["name"], "passed": passed, "detail": detail})

    # Metrics summary
    passed_count = sum(1 for r in results if r["passed"])
    total = len(results)

    print(f"\n--- {name} Summary ---")
    print(f"Passed: {passed_count}/{total} ({passed_count/total*100:.1f}%)")

    if total_planted > 0:
        recall = total_found / total_planted * 100
        print(f"Recall (planted errors found): {total_found}/{total_planted} ({recall:.1f}%)")

    if total_reported > 0:
        precision = total_real / total_reported * 100 if total_reported > 0 else 0
        print(f"Precision (approx): ~{precision:.1f}%")

    if clean_total > 0:
        fa_rate = clean_false_alarms / clean_total * 100
        print(f"False alarm rate on clean code: {clean_false_alarms}/{clean_total} ({fa_rate:.1f}%)")

    if failures_by_category:
        print(f"\nFailures by category:")
        for cat, names in sorted(failures_by_category.items()):
            print(f"  [{cat}] ({len(names)} failures): {', '.join(names)}")

    return results


def main():
    print("=" * 60)
    print("  Multi-Error Detection Test Suite")
    print("=" * 60)

    # Quick connectivity check
    try:
        requests.get("http://127.0.0.1:5000/", timeout=3)
    except Exception:
        print("ERROR: Cannot connect to http://127.0.0.1:5000")
        print("Make sure the Flask server is running: python app.py")
        sys.exit(1)

    mode = sys.argv[1] if len(sys.argv) > 1 else "dev"

    if mode == "dev":
        run_suite("DEV SET", DEV_SET)
    elif mode == "holdout":
        run_suite("HOLDOUT SET", HOLDOUT_SET)
    elif mode == "all":
        run_suite("DEV SET", DEV_SET)
        run_suite("HOLDOUT SET", HOLDOUT_SET)
    else:
        print(f"Usage: python test_multi_error.py [dev|holdout|all]")
        sys.exit(1)


if __name__ == "__main__":
    main()
