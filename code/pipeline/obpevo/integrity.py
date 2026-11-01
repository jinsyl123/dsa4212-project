"""Static integrity checks on candidate programs: keep the scoring outside the code being evolved.

`flags(src)` parses a candidate with `ast` and returns a list of rule violations. It never runs the
code and never changes a score: flagged candidates are still evaluated and logged (evals.csv
`integrity` column), so we can report how often each rule is hit instead of silently dropping them.

Rules (flag name -> what it catches):
  syntax_error      the source does not parse
  import:<mod>      any import other than numpy (os, time, sys, random, ...)
  random            np.random / numpy.random / the random module: scores must not depend on luck
  global            `global` / `nonlocal`: state shared across calls or instances
  attr_assign       assigning to an attribute (priority.cache = ..., np.x = ...) or setattr/delattr
  module_state      mutating a module-level name (cache.append(...), cache[k] = ...): memory across instances
  mutable_default   a list/dict/set default argument, the other classic way to keep state between calls
  call:<name>       exec, eval, compile, open, __import__, globals, vars, locals, input, breakpoint
  dunder:<name>     touching dunder attributes (__builtins__, __class__, __subclasses__, ...): sandbox escapes

Static checks can't catch everything (e.g. state hidden behind a numpy array view), so
`check_integrity.py` also rescores programs twice and with shuffled instances.
"""
from __future__ import annotations

import ast

ALLOWED_IMPORTS = {"numpy"}
BANNED_CALLS = {"exec", "eval", "compile", "open", "__import__", "globals", "vars", "locals", "input",
                "breakpoint", "setattr", "delattr"}
MUTATING_METHODS = {"append", "extend", "insert", "pop", "popitem", "remove", "clear", "update", "add",
                    "discard", "setdefault", "sort", "reverse", "fill", "put", "itemset", "resize"}
SAFE_DUNDERS = {"__name__", "__doc__"}


def _root_name(node: ast.AST) -> str | None:
    """`a.b[c].d` -> 'a'."""
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _module_names(tree: ast.Module) -> set[str]:
    """Names bound at module level by assignment (not functions or imports)."""
    names = set()
    for stmt in tree.body:
        targets = (stmt.targets if isinstance(stmt, ast.Assign)
                   else [stmt.target] if isinstance(stmt, (ast.AnnAssign, ast.AugAssign)) else [])
        for t in targets:
            names |= {n.id for n in ast.walk(t) if isinstance(n, ast.Name)}
    return names


def _local_names(fn: ast.FunctionDef) -> set[str]:
    """Arguments and names assigned inside a function (these shadow module-level names)."""
    a = fn.args
    names = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
    names |= {x.arg for x in (a.vararg, a.kwarg) if x}
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            names.add(n.id)
    return names


def flags(src: str) -> list[str]:
    """Sorted, de-duplicated rule violations for one program ([] = clean)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return ["syntax_error"]
    out: set[str] = set()
    numpy_aliases = {"numpy"}

    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                root = a.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    out.add(f"import:{root}")
                elif a.name.startswith("numpy.random"):
                    out.add("random")
                elif a.name == "numpy":
                    numpy_aliases.add(a.asname or "numpy")
        elif isinstance(n, ast.ImportFrom):
            root = (n.module or "").split(".")[0]
            if root not in ALLOWED_IMPORTS:
                out.add(f"import:{root or '.'}")
            elif (n.module or "").startswith("numpy.random") or any(a.name == "random" for a in n.names):
                out.add("random")
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            out.add("global")
        elif isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
            targets = n.targets if isinstance(n, (ast.Assign, ast.Delete)) else [n.target]
            if any(isinstance(t, ast.Attribute) and isinstance(t.ctx, (ast.Store, ast.Del))
                   for tgt in targets for t in ast.walk(tgt)):
                out.add("attr_assign")
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in BANNED_CALLS:
            out.add("attr_assign" if n.func.id in ("setattr", "delattr") else f"call:{n.func.id}")
        elif isinstance(n, ast.Attribute):
            if n.attr.startswith("__") and n.attr.endswith("__") and n.attr not in SAFE_DUNDERS:
                out.add(f"dunder:{n.attr}")
        elif isinstance(n, ast.Name) and n.id == "__builtins__":
            out.add("dunder:__builtins__")

    # np.random.* / numpy.random.* (the import rule already covers `import random`)
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and n.attr == "random" and _root_name(n) in numpy_aliases:
            out.add("random")

    # State kept across calls: module-level names mutated inside a function, and mutable defaults
    module_names = _module_names(tree)
    for fn in (n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))):
        if any(isinstance(d, (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp))
               for d in fn.args.defaults + [d for d in fn.args.kw_defaults if d is not None]):
            out.add("mutable_default")
        shared = module_names - _local_names(fn)
        for n in ast.walk(fn):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr in MUTATING_METHODS and _root_name(n.func.value) in shared):
                out.add("module_state")
            elif (isinstance(n, (ast.Assign, ast.AugAssign, ast.Delete))
                  and any(isinstance(t, ast.Subscript) and _root_name(t) in shared
                          for t in (n.targets if isinstance(n, (ast.Assign, ast.Delete)) else [n.target]))):
                out.add("module_state")
    return sorted(out)


def flag_str(src: str) -> str:
    """Flags as one CSV-friendly string ('' = clean)."""
    return ";".join(flags(src))
