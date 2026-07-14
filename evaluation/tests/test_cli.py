import subprocess
import sys
from pathlib import Path


def test_cli_can_run_as_a_script():
    script = Path(__file__).parents[1] / "run_evaluation.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
    assert "Evaluate final HeadStudio images" in result.stdout
