# tests/test_integration_small_bundle.py
import json
import subprocess
from pathlib import Path

def test_end_to_end(tmp_path: Path):
    # Arrange
    work = tmp_path
    (work / "input" / "track").mkdir(parents=True)
    (work / "input" / "t1").mkdir(parents=True)

    # Copy test assets (assume you commit them under tests/data/)
    small_tck = Path("tests/data/small_bundle.tck")
    ref_t1 = Path("tests/data/ref_t1.nii.gz")
    (work / "input" / "track" / "track.tck").write_bytes(small_tck.read_bytes())
    (work / "input" / "t1" / "t1.nii.gz").write_bytes(ref_t1.read_bytes())

    cfg = {
        "tck": str(work / "input/track/track.tck"),
        "t1": str(work / "input/t1/t1.nii.gz"),
        "N_points": 32,
        "perc": 0,  # disable core
        "length_thr": 0.90,
        "average_type": "mean",
        "endpoint_mode": "median",
        "smooth_density": True,
        "keep_endpoints": False,
        "representative": False
    }
    (work / "config.json").write_text(json.dumps(cfg))

    # Act
    p = subprocess.run(["bash", str(Path("main").resolve())], cwd=work, capture_output=True, text=True)

    # Assert
    assert p.returncode == 0, p.stderr + p.stdout
    assert (work / "track/track.tck").exists()
    assert (work / "product.json").exists()
