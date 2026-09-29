from __future__ import annotations

import joblib
import numpy as np

from baloto_ml.data.synthetic import synthetic_history
from baloto_ml.models.base import GameData
from baloto_ml.models.classifiers import PerNumberLogistic
from baloto_ml.models.registry import (
    METADATA_FILE,
    MODEL_FILE,
    ModelBundle,
    ModelRegistry,
    code_hash,
    version_id,
)


def make_staging(tmp_path, juego: str, version: str):
    data = GameData.from_draws(synthetic_history(60, np.random.default_rng(0)), juego)
    bundle = ModelBundle(
        juego, "logistica", {"logistica": PerNumberLogistic().fit(data, np.arange(10, 60))}
    )
    staging = tmp_path / "staging" / version
    staging.mkdir(parents=True)
    joblib.dump(bundle, staging / MODEL_FILE)
    return staging, {"version": version, "juego": juego, "datos": {"hash": version}}, data


def test_version_id_depends_only_on_inputs() -> None:
    a = version_id("2026-09-26", {"datos": "x", "codigo": "y"})
    assert a == version_id("2026-09-26", {"codigo": "y", "datos": "x"})  # orden irrelevante
    assert a != version_id("2026-09-26", {"datos": "x2", "codigo": "y"})
    assert a.startswith("2026-09-26_") and len(a) == len("2026-09-26_") + 8


def test_code_hash_ignores_line_endings(tmp_path) -> None:
    (tmp_path / "a.py").write_bytes(b"x = 1\r\ny = 2\r\n")
    h1 = code_hash(tmp_path)
    (tmp_path / "a.py").write_bytes(b"x = 1\ny = 2\n")
    assert code_hash(tmp_path) == h1
    (tmp_path / "a.py").write_bytes(b"x = 3\n")
    assert code_hash(tmp_path) != h1


def test_register_load_and_latest(tmp_path) -> None:
    reg = ModelRegistry(tmp_path / "models")
    assert reg.latest_version("baloto") is None
    staging, meta, data = make_staging(tmp_path, "baloto", "2026-01-01_aaaaaaaa")
    assert reg.register(staging, meta) == "2026-01-01_aaaaaaaa"
    assert reg.latest_version("baloto") == "2026-01-01_aaaaaaaa"
    bundle, loaded_meta = reg.load("baloto")
    assert loaded_meta == meta
    p = bundle.predict(data, np.array([60]))
    assert p.balls.shape == (1, 43)

    before = (reg.game_dir("baloto") / "2026-01-01_aaaaaaaa" / METADATA_FILE).read_bytes()
    reg.register(staging, meta | {"extra": 1})  # misma versión: no se reescribe
    after = (reg.game_dir("baloto") / "2026-01-01_aaaaaaaa" / METADATA_FILE).read_bytes()
    assert before == after


def test_prune_keeps_latest_and_newest(tmp_path) -> None:
    reg = ModelRegistry(tmp_path / "models")
    for day in range(1, 6):
        staging, meta, _ = make_staging(tmp_path, "revancha", f"2026-01-0{day}_0000000{day}")
        reg.register(staging, meta)
    removed = reg.prune("revancha", keep=2)
    assert removed == ["2026-01-01_00000001", "2026-01-02_00000002", "2026-01-03_00000003"]
    assert reg.versions("revancha") == ["2026-01-04_00000004", "2026-01-05_00000005"]
    assert reg.latest_version("revancha") == "2026-01-05_00000005"
