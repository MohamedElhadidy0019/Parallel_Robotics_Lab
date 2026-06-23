---
name: project-sim-setup-steps
description: Setup steps for cloning the shelf_gym sim repo and wiring it as a uv editable local dep
metadata:
  type: project
---

Steps to set up the sim environment (user runs these themselves):

```bash
# 1. Clone the sim repo into third_party/
git clone https://github.com/NilsDengler/manipulation_enhanced_map_prediction third_party/shelf_gym_repo

# 2. Gitignore it
echo "third_party/" >> .gitignore

# 3. Initialize uv project
uv init --no-readme

# 4. Add sim as editable local dep (enables intellisense, no sys.path hacks)
uv add --editable ./third_party/shelf_gym_repo

# 5. Sync the environment
uv sync

# 6. Verify import works
uv run python -c "from shelf_gym.environments.shelf_environment import ShelfEnv; print('OK')"
```

**Why:** third_party/ is gitignored so colleagues clone it separately. Editable install means IDE resolves shelf_gym imports through .venv — full intellisense, no sys.path.

**Known risk:** `skgeom` (scikit-geometry / CGAL binding) is imported in placement_logic_utils.py but not listed in the sim's setup.py. Will surface as a runtime import error, not during uv sync. Deal with it when step 6 fails.

**How to apply:** When continuing this setup, resume from whichever step the user left off at.
