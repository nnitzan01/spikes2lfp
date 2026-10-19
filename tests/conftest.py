import sys
from pathlib import Path

# Make the repo's top-level modules importable regardless of where pytest is launched from
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
