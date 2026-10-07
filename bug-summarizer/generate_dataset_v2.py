import os
import ast
import random
import csv
import sys
from collections import Counter

# Seed for reproducibility
random.seed(42)

CLASSES = [
    "no_bug",
    "off_by_one",
    "wrong_operator",
    "wrong_comparison",
    "missing_return",
    "wrong_variable_used",
    "wrong_loop_condition",
    "infinite_loop_risk",
    "wrong_boolean_logic",
    "mutable_default_argument",
    "string_vs_number_mix",
    "list_index_out_of_bounds",
    "unbound_local_error",
    "division_by_zero",
    "return_inside_loop_prematurely",
]

def load_clean_code():
    """Load clean code snippets from MBPP or fallback to hardcoded samples."""
    snippets = []
    try:
        from datasets import load_dataset
        print("Downloading MBPP dataset...")
        ds = load_dataset("google-research-datasets/mbpp", split="train+validation+test")
        for item in ds:
            snippets.append(item["code"])
        print(f"Loaded {len(snippets)} snippets from MBPP.")
    except Exception as e:
        print(f"Failed to load MBPP: {e}. Trying CodeSearchNet...")
        try:
            ds = load_dataset("code_search_net", "python", split="train[:5000]")
            for item in ds:
                if len(item["func_code"]) < 1000:
                    snippets.append(item["func_code"])
        except Exception as e:
            print(f"Failed to load CodeSearchNet: {e}. Falling back to standard library parsing.")
    
    if len(snippets) < 5000:
        # Fallback: scan python lib recursively
        import glob
        import sysconfig
        lib_path = sysconfig.get_path("stdlib")
        for fpath in glob.glob(os.path.join(lib_path, "**", "*.py"), recursive=True):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                tree = ast.parse(content)
                for node in tree.body:
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                        try:
                            code = ast.unparse(node)
                            if 50 < len(code) < 1000:
                                snippets.append(code)
                        except:
                            pass
            except:
                pass
        print(f"Loaded {len(snippets)} snippets from stdlib.")
    
    # Filter and clean
    valid_snippets = []
    for s in snippets:
        try:
            ast.parse(s)
            valid_snippets.append(s)
        except:
            pass
    return list(set(valid_snippets))


class BugMutator(ast.NodeTransformer):
    def __init__(self, bug_type):
        super().__init__()
        self.bug_type = bug_type
        self.mutated = False
        self.vars = []

    def visit_FunctionDef(self, node):
        if not hasattr(self, 'in_func'):
            self.in_func = True
            self.generic_visit(node)
            
            # mutable_default_argument
            if self.bug_type == "mutable_default_argument" and not self.mutated:
                if node.args.args:
                    # add mutable default to last arg
                    node.args.defaults.append(ast.List(elts=[], ctx=ast.Load()))
                    self.mutated = True
            
            self.in_func = False
        else:
            self.generic_visit(node)
        return node
        
    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store) and node.id not in self.vars:
            self.vars.append(node.id)
        
        if self.bug_type == "wrong_variable_used" and not self.mutated and isinstance(node.ctx, ast.Load):
            if self.vars and random.random() < 0.3:
                node.id = random.choice(self.vars)
                self.mutated = True
        return self.generic_visit(node)

    def visit_BinOp(self, node):
        if self.bug_type == "wrong_operator" and not self.mutated:
            ops = [ast.Add(), ast.Sub(), ast.Mult(), ast.Div(), ast.FloorDiv(), ast.Mod()]
            node.op = random.choice([op for op in ops if type(op) != type(node.op)])
            self.mutated = True
            
        elif self.bug_type == "division_by_zero" and not self.mutated:
            node.op = ast.Div()
            node.right = ast.Constant(value=0)
            self.mutated = True
            
        elif self.bug_type == "string_vs_number_mix" and not self.mutated:
            if isinstance(node.op, ast.Add):
                node.right = ast.Constant(value="1")
                self.mutated = True
                
        return self.generic_visit(node)

    def visit_Compare(self, node):
        if self.bug_type == "wrong_comparison" and not self.mutated:
            ops = [ast.Eq(), ast.NotEq(), ast.Lt(), ast.LtE(), ast.Gt(), ast.GtE()]
            node.ops = [random.choice([op for op in ops if type(op) != type(node.ops[0])])]
            self.mutated = True
            
        elif self.bug_type == "wrong_loop_condition" and hasattr(self, 'in_loop') and not self.mutated:
            node.ops = [ast.Gt() if isinstance(node.ops[0], ast.Lt) else ast.Lt()]
            self.mutated = True
            
        return self.generic_visit(node)

    def visit_BoolOp(self, node):
        if self.bug_type == "wrong_boolean_logic" and not self.mutated:
            node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
            self.mutated = True
        return self.generic_visit(node)

    def visit_Return(self, node):
        if self.bug_type == "missing_return" and not self.mutated:
            self.mutated = True
            return None # Delete return
        return self.generic_visit(node)

    def visit_For(self, node):
        self.in_loop = True
        
        if self.bug_type == "return_inside_loop_prematurely" and not self.mutated:
            node.body.append(ast.Return(value=ast.Constant(value=True)))
            self.mutated = True
            
        self.generic_visit(node)
        self.in_loop = False
        return node
        
    def visit_While(self, node):
        self.in_loop = True
        if self.bug_type == "infinite_loop_risk" and not self.mutated:
            node.test = ast.Constant(value=True)
            self.mutated = True
        self.generic_visit(node)
        self.in_loop = False
        return node
        
    def visit_Call(self, node):
        if self.bug_type == "off_by_one" and not self.mutated:
            if isinstance(node.func, ast.Name) and node.func.id == "range":
                if len(node.args) == 1:
                    node.args[0] = ast.BinOp(left=node.args[0], op=ast.Add(), right=ast.Constant(value=1))
                    self.mutated = True
                elif len(node.args) >= 2:
                    node.args[1] = ast.BinOp(left=node.args[1], op=ast.Sub(), right=ast.Constant(value=1))
                    self.mutated = True
        return self.generic_visit(node)
        
    def visit_Subscript(self, node):
        if self.bug_type == "list_index_out_of_bounds" and not self.mutated:
            if isinstance(node.slice, ast.Name):
                node.slice = ast.BinOp(left=node.slice, op=ast.Add(), right=ast.Constant(value=100))
                self.mutated = True
            elif isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int):
                node.slice.value += 100
                self.mutated = True
        return self.generic_visit(node)


def apply_mutation(code, bug_type):
    if bug_type == "no_bug":
        return code, True
        
    try:
        tree = ast.parse(code)
    except:
        return code, False
        
    mutator = BugMutator(bug_type)
    mutated_tree = mutator.visit(tree)
    
    if mutator.mutated:
        try:
            ast.fix_missing_locations(mutated_tree)
            mutated_code = ast.unparse(mutated_tree)
            return mutated_code, True
        except:
            return code, False
            
    # Fallback string mutations if AST failed
    if not mutator.mutated:
        if bug_type == "unbound_local_error":
            return code + "\n    print(undeclared_local_var)", True
        elif bug_type == "missing_return":
            if "return " in code:
                return code.replace("return ", "result = "), True
        elif bug_type == "division_by_zero":
            return code + "\n    bad_div = 1 / 0", True
        elif bug_type == "off_by_one":
            return code + "\n    for i in range(len(locals()) + 1):\n        pass", True
        elif bug_type == "wrong_operator":
            if "+" in code: return code.replace("+", "-"), True
            elif "-" in code: return code.replace("-", "+"), True
        elif bug_type == "string_vs_number_mix":
            return code + "\n    mix = 'hello' + 5", True
        elif bug_type == "wrong_boolean_logic":
            if " and " in code: return code.replace(" and ", " or "), True
            elif " or " in code: return code.replace(" or ", " and "), True
        elif bug_type == "wrong_comparison":
            if "==" in code: return code.replace("==", "!="), True
            elif "!=" in code: return code.replace("!=", "=="), True
            elif "<" in code: return code.replace("<", ">"), True
        elif bug_type == "infinite_loop_risk":
            return code + "\n    while True:\n        pass", True
        elif bug_type == "wrong_loop_condition":
            return code + "\n    n = 10\n    while n < 0:\n        n -= 1", True
        elif bug_type == "list_index_out_of_bounds":
            return code + "\n    arr = [1]\n    bad = arr[5]", True
        elif bug_type == "return_inside_loop_prematurely":
            return code + "\n    for _ in range(5):\n        return False", True
        elif bug_type == "wrong_variable_used":
            return code + "\n    temp_a = 1\n    temp_b = temp_a", True
            
    return code, False


def generate():
    snippets = load_clean_code()
    if not snippets:
        print("No snippets found.")
        return
        
    # Create dataset
    dataset = []
    
    # We want 15000-30000 samples.
    # Each snippet can generate multiple bugs.
    
    # Create dataset balancing loop
    print(f"Generating dataset from {len(snippets)} clean snippets...")
    
    target_per_class = 1500
    class_counts = Counter()
    dataset = []
    
    # 1. First add no_bug
    for i, code in enumerate(snippets):
        r = i % 10
        split = "train" if r < 8 else ("val" if r == 8 else "test")
        if class_counts["no_bug"] < target_per_class:
            dataset.append({"code": code, "bug_type": "no_bug", "split": split})
            class_counts["no_bug"] += 1
            
    # 2. Add mutated bugs
    for bug in CLASSES:
        if bug == "no_bug": continue
        
        attempts = 0
        while class_counts[bug] < target_per_class and attempts < len(snippets) * 3:
            code = random.choice(snippets)
            r = hash(code) % 10
            split = "train" if r < 8 else ("val" if r == 8 else "test")
            
            mutated, success = apply_mutation(code, bug)
            if success and mutated != code:
                dataset.append({"code": mutated, "bug_type": bug, "split": split})
                class_counts[bug] += 1
            attempts += 1
            
    # Balance classes
    print(f"Total initial generated: {len(dataset)}")
    
    # We can write to CSV
    os.makedirs("data", exist_ok=True)
    with open("data/bug_dataset.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["code", "bug_type", "split"])
        writer.writeheader()
        
        counts = Counter()
        for row in dataset:
            writer.writerow(row)
            counts[row['bug_type']] += 1
            
    print("Dataset saved. Distribution:")
    for k, v in counts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    generate()
