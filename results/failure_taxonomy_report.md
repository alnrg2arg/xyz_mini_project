# Failure Taxonomy Report

## Failure type frequency

- **slow_progress**: 478
- **unknown**: 122

## Stage frequency

- **approach**: 466
- **reach**: 89
- **align**: 45

## Examples (up to 5 each)


### unknown

- E0000 t=0 | dataset_v2/frames/E0000/000000.png
  - note: The gripper is moving toward the target position.
- E0000 t=10 | dataset_v2/frames/E0000/000010.png
  - note: The gripper is moving towards the target but the exact progress is unclear.
- E0000 t=16 | dataset_v2/frames/E0000/000016.png
  - note: The gripper is moving toward the target but is not aligned yet.
- E0001 t=0 | dataset_v2/frames/E0001/000000.png
  - note: The robot is moving towards the target position.
- E0002 t=0 | dataset_v2/frames/E0002/000000.png
  - note: The gripper is moving towards the target but its exact position relative to the target is unclear.

### slow_progress

- E0000 t=6 | dataset_v2/frames/E0000/000006.png
  - note: The gripper is close to the target but not moving significantly closer.
- E0000 t=22 | dataset_v2/frames/E0000/000022.png
  - note: The gripper is moving slowly towards the target position.
- E0000 t=26 | dataset_v2/frames/E0000/000026.png
  - note: The gripper is moving slowly towards the target.
- E0000 t=32 | dataset_v2/frames/E0000/000032.png
  - note: The gripper is slowly moving toward the target.
- E0000 t=38 | dataset_v2/frames/E0000/000038.png
  - note: The gripper is close to the target but has not moved significantly this timestep.