from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "biped.xml"
PLAYBACK_SPEED = 1.5
TRAJECTORY_SAMPLE_EVERY_STEPS = 10
TRAJECTORY_FLUSH_EVERY_STEPS = 100

