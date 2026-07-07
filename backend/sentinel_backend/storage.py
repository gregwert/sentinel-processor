"""Session storage layer: per-session filesystem layout, artifact I/O, and TTL reaping."""
import json, os, time, uuid
from pathlib import Path
import numpy as np
from sentinel_backend.models import GeoMeta, ChipGridSpec

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
SESSION_TTL = int(os.environ.get("SESSION_TTL_HOURS", "6")) * 3600


class SessionStorage:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.root = DATA_DIR / "sessions" / session_id
        self.root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(cls) -> "SessionStorage":
        s = cls(str(uuid.uuid4()))
        info = {"created_at": time.time(), "expires_at": time.time() + SESSION_TTL}
        (s.root / "session.json").write_text(json.dumps(info))
        return s

    @classmethod
    def get(cls, session_id: str) -> "SessionStorage":
        s = cls(session_id)
        if not (s.root / "session.json").exists():
            raise KeyError(f"Session {session_id} not found")
        return s

    @classmethod
    def delete(cls, session_id: str):
        import shutil
        path = DATA_DIR / "sessions" / session_id
        if path.exists():
            shutil.rmtree(path)

    def info(self) -> dict:
        return json.loads((self.root / "session.json").read_text())

    # --- artifact helpers ---
    def save_array(self, name: str, arr: np.ndarray):
        np.save(str(self.root / f"{name}.npy"), arr)

    def load_array(self, name: str) -> np.ndarray:
        path = self.root / f"{name}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Artifact '{name}' not found in session {self.session_id}")
        return np.load(str(path))

    def artifact_exists(self, name: str) -> bool:
        return (self.root / f"{name}.npy").exists()

    def stages_ready(self) -> list[str]:
        return [n for n in ["stretched", "dehazed", "enhanced"] if self.artifact_exists(n)]

    # --- meta helpers ---
    def save_meta(self, meta: GeoMeta):
        (self.root / "meta.json").write_text(meta.model_dump_json())

    def load_meta(self) -> GeoMeta:
        return GeoMeta.model_validate_json((self.root / "meta.json").read_text())

    # --- grid helpers ---
    def save_grid(self, spec: ChipGridSpec):
        (self.root / "grid.json").write_text(spec.model_dump_json())

    def load_grid(self) -> ChipGridSpec:
        return ChipGridSpec.model_validate_json((self.root / "grid.json").read_text())

    def grid_exists(self) -> bool:
        return (self.root / "grid.json").exists()

    # --- reference stats helpers ---
    def save_reference_stats(self, stats: dict):
        # stats has keys: n (int), cdfs (list of 3 arrays), mean (array), std (array)
        np.savez(str(self.root / "reference_stats.npz"),
                 n=np.array([stats["n"]]),
                 cdf0=stats["cdfs"][0], cdf1=stats["cdfs"][1], cdf2=stats["cdfs"][2],
                 mean=stats["mean"], std=stats["std"])

    def load_reference_stats(self) -> dict | None:
        path = self.root / "reference_stats.npz"
        if not path.exists():
            return None
        d = np.load(str(path))
        return {"n": int(d["n"][0]),
                "cdfs": [d["cdf0"], d["cdf1"], d["cdf2"]],
                "mean": d["mean"], "std": d["std"]}

    def reference_stats_exist(self) -> bool:
        return (self.root / "reference_stats.npz").exists()

    # --- source TIFF helpers ---
    def source_tiff_path(self) -> Path:
        return self.root / "source.tif"

    def ref_tiff_dir(self) -> Path:
        d = self.root / "refs"
        d.mkdir(exist_ok=True)
        return d

    # --- export helpers ---
    def export_dir(self) -> Path:
        d = self.root / "exports"
        d.mkdir(exist_ok=True)
        return d

    def export_path(self, job_id: str) -> Path:
        return self.export_dir() / f"{job_id}.zip"

    # --- chip stats helpers (filter results) ---
    def save_chip_stats(self, accepted: list[int], rejected: list[int], stats: list[dict]):
        data = {"accepted": accepted, "rejected": rejected, "stats": stats}
        (self.root / "chip_stats.json").write_text(json.dumps(data))

    def load_chip_stats(self) -> dict | None:
        path = self.root / "chip_stats.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def chip_stats_exist(self) -> bool:
        return (self.root / "chip_stats.json").exists()


def reap_expired_sessions():
    """Delete session directories whose TTL has passed. Call from a background task."""
    sessions_dir = DATA_DIR / "sessions"
    if not sessions_dir.exists():
        return
    now = time.time()
    for session_dir in sessions_dir.iterdir():
        try:
            info = json.loads((session_dir / "session.json").read_text())
            if now > info["expires_at"]:
                import shutil
                shutil.rmtree(session_dir)
        except Exception:
            pass
