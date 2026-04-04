# Environment Configuration Guide (BabyAI/MiniGrid)

This guide is focused on what you asked for: environment kwargs you can sweep over to control horizon/complexity, especially object-composition and sequential tasks.

## 1) Where Environment Config Is Wired In This Repo

Environment selection/kwargs flow through these files:

1. `examples/train/babyai_text/run_babyai_text.sh`
   - Launch entrypoint.
   - This is where you set the environment ID and pass config overrides for training runs/sweeps.

2. `skyrl/train/config/ppo_base_config.yaml`
   - Default training config.
   - Holds default env-related config keys (env name and env kwargs container).

3. `skyrl/train/trainer.py`
   - Runtime wiring.
   - Reads env config and constructs the actual Gymnasium/MiniGrid environment (kwargs are unpacked here into env creation).

If you want parameter sweeps, this is the practical pattern:
- Keep one base config in `ppo_base_config.yaml`.
- Override env ID + env kwargs from `run_babyai_text.sh` (or your Slurm loop).
- Run one job per `(env_id, env_kwargs)` point.

## 2) Sweepable BabyAI/MiniGrid Environment Kwargs

These are the kwargs exposed by parameterized BabyAI environment families in docs/specs.

| Kwarg | What it controls | Typical effect on difficulty/horizon |
|---|---|---|
| `room_size` | Grid size of room(s) | Longer paths, larger search space |
| `num_dists` | Number of distractor objects | Harder grounding/disambiguation |
| `num_rows` | Number of room rows in multi-room layouts | More exploration and navigation depth |
| `num_cols` | Number of room columns in multi-room layouts | More exploration and branching |
| `num_objs` | Number of objects in local manipulation variants | More object-selection complexity |
| `objs_per_room` | Objects per room in multi-room manipulation variants | More clutter, more wrong-object risk |
| `num_doors` | Doors/tasks in ordered-door tasks | Longer sequential subtask chains |

## 3) Environment Families: Name -> kwargs and values

Values below are from official registered configs and environment specs.

| Environment family / examples | kwargs | Values seen in official configs/specs |
|---|---|---|
| `BabyAI-GoToLocalSxNy-v0` (examples: `GoToLocalS5N2`, `GoToLocalS8N7`) | `room_size`, `num_dists` | `room_size` in `{5,6,7,8}`, `num_dists` in `{1..10}` |
| `BabyAI-GoToObjSx-v0` (examples: `GoToObjS4`, `GoToObjS8`) | `room_size` | `room_size` in `{4,5,6,7,8}` |
| `BabyAI-GoToObjMazeSxRy-v0` (example: `GoToObjMazeS4R2`) | `room_size`, `num_rows`, `num_cols` | Observed: `room_size=4`, `num_rows=2`, `num_cols=2` (`S4R2`) |
| `BabyAI-GoToSeqSxRy-v0` (example: `GoToSeqS5R2`) | `room_size`, `num_rows`, `num_cols`, `num_dists` | Observed: `room_size=5`, `num_rows=2`, `num_cols=2`, `num_dists=4` |
| `BabyAI-KeyCorridorSxRy-v0` (examples: `S3R1`, `S6R3`) | `room_size`, `num_rows` | `room_size` in `{3,4,5,6}`, `num_rows` in `{1,2,3}` |
| `BabyAI-PutNextLocalSxNy-v0` (example: `PutNextLocalS6N4`) | `room_size`, `num_objs` | Observed: `room_size=6`, `num_objs=4` |
| `BabyAI-PutNextSxNy-v0` (examples: `PutNextS5N2`, `PutNextS7N4`) | `room_size`, `objs_per_room` | `room_size` in `{5,6,7}`, `objs_per_room` in `{2,3,4}` |
| `BabyAI-MoveTwoAcrossSxNy-v0` (examples: `MoveTwoAcrossS5N2`, `MoveTwoAcrossS8N9`) | `room_size`, `objs_per_room` | Observed configs include `S5N2`, `S8N1`, `S8N9..S8N16` |
| `BabyAI-OpenDoorsOrderNn-v0` (examples: `OpenDoorsOrderN2`, `OpenDoorsOrderN4`) | `num_doors` | `num_doors` in `{2,4}` |
| `BabyAI-FindObjSx-v0` (examples: `FindObjS5`, `FindObjS6`) | `room_size` | Observed: `room_size` includes at least `{5,6}` (and `S7` appears in docs family variants) |

### Environments with no explicit kwargs in published specs (default constructor behavior)

For the following IDs, published env specs show empty kwargs (`{}`), so complexity is primarily from the fixed task design:

- `BabyAI-Synth-v0`
- `BabyAI-SynthSeq-v0`
- `BabyAI-SynthLoc-v0`
- `BabyAI-Pickup-v0`
- `BabyAI-PickupLoc-v0`
- `BabyAI-UnlockPickup-v0`
- `BabyAI-BossLevelNoUnlock-v0`
- `BabyAI-ActionObjDoor-v0`
- `BabyAI-OpenRedDoor-v0`
- `BabyAI-GoToSeq-v0` (base)

## 4) Best Environment Families for Your Main Goal (horizon/composition sweeps)

For “pick object A then object B” / increasing chain length:

1. `GoToSeq` family
   - Use `num_rows/num_cols` and `num_dists` to increase chain + search complexity.

2. `OpenDoorsOrderNn`
   - Direct knob for sequential subtasks via `num_doors`.

3. `PutNext` / `MoveTwoAcross`
   - Object manipulation with clutter (`objs_per_room`) and larger space (`room_size`).

4. `KeyCorridor`
   - Navigation + dependency structure; sweep `num_rows` and `room_size`.

## 5) Practical Sweep Axes (minimal + useful)

Recommended sweep dimensions:

1. Navigation horizon:
   - `room_size`: small -> medium -> large
2. Distractor load:
   - `num_dists` or `objs_per_room`: low -> medium -> high
3. Structural depth:
   - `num_rows`, `num_cols`, `num_doors`: shallow -> deeper sequential requirements

## 6) Notes on Interpretation

- Some values above are directly explicit in published kwargs.
- Some value ranges are inferred from official registered environment naming patterns (`Sx`, `Ny`, `Ry`) when specs list multiple registered IDs.
- If you want truly arbitrary values (beyond registered IDs), you can instantiate MiniGrid env classes directly with kwargs in code instead of relying only on pre-registered IDs.

## 7) Sources

- MiniGrid BabyAI docs:
  - https://minigrid.farama.org/environments/babyai/GoTo/
  - https://minigrid.farama.org/environments/babyai/GoToLocal/
  - https://minigrid.farama.org/environments/babyai/GoToObj/
  - https://minigrid.farama.org/environments/babyai/GoToObjMaze/
  - https://minigrid.farama.org/environments/babyai/GoToSeq/
  - https://minigrid.farama.org/environments/babyai/KeyCorridor/
  - https://minigrid.farama.org/environments/babyai/PutNext/
  - https://minigrid.farama.org/environments/babyai/PutNextLocal/
  - https://minigrid.farama.org/environments/babyai/MoveTwoAcross/
  - https://minigrid.farama.org/environments/babyai/OpenDoorsOrder/
  - https://minigrid.farama.org/environments/babyai/FindObj/
- Minari environment specs/examples:
  - https://minari.farama.org/datasets/minigrid/
  - https://minari.farama.org/datasets/minigrid/BabyAI-GoToLocalS8N7-v0/
  - https://minari.farama.org/datasets/minigrid/BabyAI-GoToLocalS5N2-v0/
  - https://minari.farama.org/datasets/minigrid/BabyAI-GoToSeqS5R2-v0/
  - https://minari.farama.org/datasets/minigrid/BabyAI-KeyCorridorS6R3-v0/
  - https://minari.farama.org/datasets/minigrid/BabyAI-PutNextLocalS6N4-v0/
  - https://minari.farama.org/datasets/minigrid/BabyAI-PutNextS7N4-v0/
  - https://minari.farama.org/datasets/minigrid/BabyAI-MoveTwoAcrossS5N2-v0/
  - https://minari.farama.org/datasets/minigrid/BabyAI-OpenDoorsOrderN2-v0/
  - https://minari.farama.org/datasets/minigrid/BabyAI-FindObjS6-v0/
