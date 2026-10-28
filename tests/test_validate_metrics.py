# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""The custom-metrics validator: its result, its exit code, and no work on import."""

import contextlib
import importlib
import json
import os
import signal
import sys
import time

import pytest

from metric_library import validate_metrics

# A module whose import starts a process, records its id and then hangs.
SPAWNER = """\
import pathlib
import subprocess
import sys
import time

started = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
pathlib.Path(__file__).with_name("started.pid").write_text(str(started.pid))
time.sleep(60)
"""

# A module whose import starts a process in a new session, which keeps the
# importing process's output open, and returns at once.
DETACHER = """\
import pathlib
import subprocess
import sys

started = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
)
pathlib.Path(__file__).with_name("started.pid").write_text(str(started.pid))
"""


def _validate(monkeypatch, capsys, tmp_path, source):
    """Validate ``source`` as a plugin file; return the exit code and the result."""
    plugin = tmp_path / "custom_metrics.py"
    plugin.write_text(source, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["validate_metrics", str(plugin)])
    code = validate_metrics.main()
    return code, json.loads(capsys.readouterr().out)


def test_importing_the_module_checks_no_file(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["tool"])

    importlib.reload(validate_metrics)

    assert capsys.readouterr().out == ""


def test_a_file_whose_imports_resolve_passes(monkeypatch, capsys, tmp_path):
    source = "import json\nfrom os import path\n\n\ndef f():\n    return path\n"

    assert _validate(monkeypatch, capsys, tmp_path, source) == (
        0,
        {"ok": True, "errors": []},
    )


def test_a_syntax_error_fails(monkeypatch, capsys, tmp_path):
    code, result = _validate(monkeypatch, capsys, tmp_path, "def broken(:\n")

    assert code == 1
    assert result["ok"] is False
    assert len(result["errors"]) == 1


def test_an_import_that_does_not_resolve_fails(monkeypatch, capsys, tmp_path):
    source = "import json\nimport no_such_module_for_the_validator\n"

    code, result = _validate(monkeypatch, capsys, tmp_path, source)

    assert code == 1
    assert result["ok"] is False
    assert result["errors"][0].endswith(
        "ModuleNotFoundError: No module named 'no_such_module_for_the_validator'"
    )


def test_an_optional_import_with_a_fallback_passes(monkeypatch, capsys, tmp_path):
    """The plugin loads without the optional module, so the check passes too."""
    source = (
        "try:\n"
        "    import no_such_module_for_the_validator as fast_json\n"
        "except ImportError:\n"
        "    import json as fast_json\n"
    )

    assert _validate(monkeypatch, capsys, tmp_path, source) == (
        0,
        {"ok": True, "errors": []},
    )


def test_an_import_for_type_checkers_only_passes(monkeypatch, capsys, tmp_path):
    source = (
        "from typing import TYPE_CHECKING\n"
        "\n"
        "if TYPE_CHECKING:\n"
        "    from no_such_module_for_the_validator import Thing\n"
    )

    assert _validate(monkeypatch, capsys, tmp_path, source) == (
        0,
        {"ok": True, "errors": []},
    )


@pytest.mark.parametrize(
    "source",
    [
        "try:\n"
        "    import no_such_module_for_the_validator\n"
        "except ValueError:\n"
        "    pass\n",
        "def compute():\n    import no_such_module_for_the_validator\n",
        "try:\n"
        "    import no_such_module_for_the_validator\n"
        "except ImportError as error:\n"
        '    raise ImportError("install it") from error\n',
        "try:\n"
        "    import no_such_module_for_the_validator\n"
        "except ImportError:\n"
        '    print("missing")\n'
        "    raise\n",
        "try:\n"
        "    import no_such_module_for_the_validator\n"
        "except ModuleNotFoundError:\n"
        '    sys.exit("install it")\n',
        "def _require(name):\n"
        '    raise ImportError(f"install {name}")\n'
        "\n"
        "\n"
        "try:\n"
        "    import no_such_module_for_the_validator\n"
        "except ImportError:\n"
        '    _require("no_such_module_for_the_validator")\n',
        "def _fail(message):\n"
        "    raise ImportError(message)\n"
        "\n"
        "\n"
        "def _require(name):\n"
        '    _fail(f"install {name}")\n'
        "\n"
        "\n"
        "try:\n"
        "    import no_such_module_for_the_validator\n"
        "except ImportError:\n"
        '    _require("no_such_module_for_the_validator")\n',
        "try:\n"
        "    import no_such_module_for_the_validator\n"
        "except ImportError:\n"
        "    os.abort()\n",
    ],
    ids=[
        "try-without-an-import-error-handler",
        "inside-a-function",
        "handler-raises-a-new-error",
        "handler-raises-again",
        "handler-exits",
        "handler-calls-a-helper-that-raises",
        "handler-calls-a-helper-that-calls-one-that-raises",
        "handler-aborts",
    ],
)
def test_other_imports_are_still_checked(monkeypatch, capsys, tmp_path, source):
    code, result = _validate(monkeypatch, capsys, tmp_path, source)

    assert code == 1
    assert result["errors"][0].endswith(
        "No module named 'no_such_module_for_the_validator'"
    )


@pytest.mark.parametrize(
    "handlers",
    [
        "except ModuleNotFoundError:\n    pass\nexcept ImportError:\n    raise\n",
        "except ModuleNotFoundError:\n    no_such_name_for_the_validator = None\n",
    ],
    ids=["import-error-handler-behind-raises", "no-handler-behind"],
)
def test_a_missing_name_is_not_caught_by_a_module_not_found_handler(
    monkeypatch, capsys, tmp_path, handlers
):
    """A missing name raises a plain ImportError, not a ModuleNotFoundError."""
    source = "try:\n    from json import no_such_name_for_the_validator\n" + handlers

    code, result = _validate(monkeypatch, capsys, tmp_path, source)

    assert code == 1
    assert "cannot import name 'no_such_name_for_the_validator'" in result["errors"][0]


def test_a_missing_module_that_a_module_not_found_handler_takes_passes(
    monkeypatch, capsys, tmp_path
):
    """The ImportError handler behind it raises only for a missing name."""
    source = (
        "try:\n"
        "    import no_such_module_for_the_validator\n"
        "except ModuleNotFoundError:\n"
        "    no_such_module_for_the_validator = None\n"
        "except ImportError:\n"
        "    raise\n"
    )

    assert _validate(monkeypatch, capsys, tmp_path, source) == (
        0,
        {"ok": True, "errors": []},
    )


@pytest.mark.parametrize(
    "handler",
    [
        '    def scorer(*args):\n        raise ImportError("install it")\n',
        "    try:\n"
        "        import json as scorer\n"
        "    except ImportError:\n"
        "        raise\n",
    ],
    ids=["defines-a-stub-that-raises-when-called", "inner-fallback-try"],
)
def test_a_handler_that_goes_on_without_the_module_passes(
    monkeypatch, capsys, tmp_path, handler
):
    """The handler's raise runs only later, or only if the fallback fails too."""
    source = (
        "try:\n"
        "    import no_such_module_for_the_validator as scorer\n"
        "except ImportError:\n" + handler
    )

    assert _validate(monkeypatch, capsys, tmp_path, source) == (
        0,
        {"ok": True, "errors": []},
    )


@pytest.mark.parametrize(
    ("helper", "call"),
    [
        (
            "def _unavailable(name):\n"
            "    def scorer(*args):\n"
            '        raise ImportError(f"install {name}")\n'
            "\n"
            "    return scorer\n",
            'scorer = _unavailable("no_such_module_for_the_validator")',
        ),
        (
            "def _note(message):\n"
            "    if not message:\n"
            '        raise ValueError("empty message")\n'
            "    print(message)\n",
            '_note("no_such_module_for_the_validator is not installed")',
        ),
    ],
    ids=["returns-a-stub-that-raises-when-called", "raises-only-for-a-bad-argument"],
)
def test_a_handler_that_calls_a_helper_that_returns_passes(
    monkeypatch, capsys, tmp_path, helper, call
):
    source = (
        f"{helper}\n\n"
        "try:\n"
        "    from no_such_module_for_the_validator import scorer\n"
        "except ImportError:\n"
        f"    {call}\n"
    )

    assert _validate(monkeypatch, capsys, tmp_path, source) == (
        0,
        {"ok": True, "errors": []},
    )


def _running(pid):
    """Whether ``pid`` still runs after a grace period for its reaping."""
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        time.sleep(0.1)
    return True


def test_hanging_imports_stop_with_the_processes_they_started(
    monkeypatch, capsys, tmp_path
):
    (tmp_path / "spawner.py").write_text(SPAWNER, encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))

    try:
        result = _validate(monkeypatch, capsys, tmp_path, "import spawner\n")

        assert result == (
            1,
            {"ok": False, "errors": ["The imports did not finish within 5 seconds"]},
        )
        assert not _running(int((tmp_path / "started.pid").read_text()))
    finally:
        with contextlib.suppress(FileNotFoundError, ProcessLookupError):
            os.kill(int((tmp_path / "started.pid").read_text()), signal.SIGKILL)


def test_imports_that_leave_a_process_holding_the_output_time_out_cleanly(
    monkeypatch, capsys, tmp_path
):
    """Only the exited child is left in its group; the report still comes out."""
    (tmp_path / "detacher.py").write_text(DETACHER, encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))

    try:
        assert _validate(monkeypatch, capsys, tmp_path, "import detacher\n") == (
            1,
            {"ok": False, "errors": ["The imports did not finish within 5 seconds"]},
        )
    finally:
        with contextlib.suppress(FileNotFoundError, ProcessLookupError):
            os.kill(int((tmp_path / "started.pid").read_text()), signal.SIGKILL)
