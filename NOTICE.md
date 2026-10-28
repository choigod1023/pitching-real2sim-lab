# Data attribution and license notice

This project uses data from the **OpenBiomechanics Project (OBP)**, created by
Driveline Baseball Research and Development: https://github.com/drivelineresearch/openbiomechanics
and https://openbiomechanics.org

- The OBP data and documentation are licensed under **CC BY-NC-SA 4.0**
  (https://creativecommons.org/licenses/by-nc-sa/4.0/).
- Files in this repository that are derived from OBP data (for example the `.npz`
  files under `artifacts/`, the result JSON and figures under `marker_audit/`, and the
  tracked ball positions and results under `video_track/results/`, which come from the
  OBP-CV camera videos and Theia3D output of `OBP_movement_BaseballThrow_001`)
  are shared under the **same CC BY-NC-SA 4.0 license**. Changes were made: marker
  and joint signals were filtered, resampled, fitted to models, or summarized, and ball
  positions were measured in the videos. Use is non-commercial only.
- Under the OBP license, an employee, contractor, associate, or significant
  shareholder of a professional sports organization or financial analysis firm may
  not use OBP data or derived works for any use whatsoever without a separate
  written commercial license from Driveline. Read the full terms at
  https://github.com/drivelineresearch/openbiomechanics/blob/main/LICENSE-DATA.md
- Raw C3D files and OBP-CV videos are **not** included. Download them from the OBP
  sources and follow their terms.
- The source code written for this project is licensed under the MIT License (see `LICENSE`).

This project is not endorsed by Driveline Baseball.
