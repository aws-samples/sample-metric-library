#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Validate Python syntax and imports in a source file for custom metrics."""

import ast
import json
import os
import signal
import subprocess
import sys


# The two ways an import fails: a missing module raises ModuleNotFoundError,
# and a missing name a plain ImportError, which ``except ModuleNotFoundError``
# does not catch. A ``try`` around imports can go on after one and stop at the
# other.
_FAILURES = ("ModuleNotFoundError", "ImportError")


def _checked_imports(nodes, helpers: frozenset[str]) -> list[ast.stmt]:
    """The statements that run the imports among ``nodes`` that must resolve.

    In source order; ``helpers`` come from ``_stopping_helpers``. Left out: the
    imports under ``if TYPE_CHECKING:``, which never run, and the imports in a
    ``try`` body that the plugin goes on without, whichever way they fail
    (``_failures_survived``). When it goes on after only one of the failures,
    the imports run in a ``try`` that lets that one pass and raises the other.
    """
    checked = []
    for node in nodes:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            checked.append(node)
        elif isinstance(node, ast.If) and "TYPE_CHECKING" in (
            getattr(node.test, "id", None),
            getattr(node.test, "attr", None),
        ):
            checked += _checked_imports(node.orelse, helpers)
        elif isinstance(node, (ast.Try, ast.TryStar)):
            survived = _failures_survived(node.handlers, helpers)
            if len(survived) < len(_FAILURES):
                body = _checked_imports(node.body, helpers)
                if survived and body:
                    body = [
                        ast.Try(
                            body=body,
                            handlers=[
                                ast.ExceptHandler(
                                    type=ast.Name(id=failure, ctx=ast.Load()),
                                    name=None,
                                    body=[
                                        ast.Pass()
                                        if failure in survived
                                        else ast.Raise(exc=None, cause=None)
                                    ],
                                )
                                for failure in _FAILURES
                            ],
                            orelse=[],
                            finalbody=[],
                        )
                    ]
                checked += body
            checked += _checked_imports(
                node.handlers + node.orelse + node.finalbody, helpers
            )
        else:
            # Import statements sit only in the statements of a block.
            checked += _checked_imports(
                (
                    child
                    for child in ast.iter_child_nodes(node)
                    if isinstance(child, (ast.stmt, ast.excepthandler, ast.match_case))
                ),
                helpers,
            )
    return checked


def _failures_caught(handler: ast.ExceptHandler) -> set[str]:
    """The import failures (``_FAILURES``) that ``handler`` catches.

    A bare handler, or one naming ``ImportError`` or a base of it, catches
    both; one naming only ``ModuleNotFoundError`` catches a missing module.
    """
    if handler.type is None:
        return set(_FAILURES)
    caught = (
        handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    )
    names = {getattr(name, "id", getattr(name, "attr", None)) for name in caught}
    if names & {"ImportError", "Exception", "BaseException"}:
        return set(_FAILURES)
    return names & {"ModuleNotFoundError"}


def _stops(node: ast.AST, helpers: frozenset[str] = frozenset()) -> bool:
    """Whether running ``node`` raises or ends the process.

    That is a ``raise``, or a call of a function named ``exit``, ``quit``,
    ``_exit`` or ``abort``, such as ``sys.exit`` or ``os.abort``, or of one of
    ``helpers``. The bodies of nested functions and the handlers of inner
    ``try`` blocks do not count: they run later, or only when the inner
    ``try`` fails.
    """
    pending = [node]
    while pending:
        current = pending.pop()
        if isinstance(current, ast.Raise):
            return True
        if isinstance(current, ast.Call):
            name = getattr(current.func, "id", getattr(current.func, "attr", None))
            if name in ("exit", "quit", "_exit", "abort") or name in helpers:
                return True
        pending.extend(
            child
            for child in ast.iter_child_nodes(current)
            if not isinstance(
                child,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ExceptHandler),
            )
        )
    return False


def _stopping_helpers(tree: ast.AST) -> frozenset[str]:
    """The module-level functions of ``tree`` that always raise or exit.

    A function counts when a statement of its own body, not one inside an
    ``if``, a loop or another block, raises, exits or calls a function that
    counts (``_stops``), through any number of such calls.
    """
    functions = [
        node
        for node in getattr(tree, "body", [])
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    helpers = frozenset()
    while True:
        found = frozenset(
            function.name
            for function in functions
            if any(
                isinstance(statement, (ast.Raise, ast.Expr, ast.Return))
                and _stops(statement, helpers)
                for statement in function.body
            )
        )
        if found == helpers:
            return helpers
        helpers = found


def _failures_survived(
    handlers: list[ast.ExceptHandler], helpers: frozenset[str]
) -> set[str]:
    """The import failures of a ``try`` body that the plugin goes on after.

    Each failure counts at the first of ``handlers`` that catches it: the
    plugin goes on when that handler does not raise or exit (``_stops``, with
    ``helpers``). A failure that no handler catches stops the plugin.
    """
    goes_on = {}
    for handler in handlers:
        for failure in _failures_caught(handler):
            goes_on.setdefault(failure, not _stops(handler, helpers))
    return {failure for failure, survived in goes_on.items() if survived}


def main() -> int:
    """Check the file named by the first argument; print the result, return the exit code."""
    file = sys.argv[1]
    with open(file) as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(json.dumps({"ok": False, "errors": [str(e)]}))
        return 1

    checked = _checked_imports(tree.body, _stopping_helpers(tree))
    imports = ast.unparse(ast.Module(body=checked, type_ignores=[]))
    if imports:
        # No shell; -c runs only the checked file's imports, rebuilt by
        # ast.unparse, some in a try that lets a failure the plugin survives pass.
        with subprocess.Popen(  # nosec B603 # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
            [sys.executable, "-c", imports],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        ) as child:
            try:
                _, stderr = child.communicate(timeout=5)
            except subprocess.TimeoutExpired as error:
                # The child leads its own process group on POSIX, so killing
                # the group also stops the processes that the imports started
                # in it. Windows has no process groups, and macOS refuses to
                # signal a group whose only member is the exited child.
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except (AttributeError, ProcessLookupError, PermissionError):
                    child.kill()
                message = f"The imports did not finish within {error.timeout} seconds"
                print(json.dumps({"ok": False, "errors": [message]}))
                return 1
        if child.returncode != 0:
            print(json.dumps({"ok": False, "errors": [stderr.strip()]}))
            return 1

    print(json.dumps({"ok": True, "errors": []}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
