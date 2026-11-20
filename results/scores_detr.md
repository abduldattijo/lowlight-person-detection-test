# Scores (facebook/detr-resnet-50, person, threshold 0.5, IoU 0.5)

Recall = matched visible person boxes / visible person boxes. Dropouts = person matched in one frame and missed in the next (eligible = visible in both frames, same track ID). Miss rate is shown next to dropouts because the arms have different denominators.

| Clip | Arm | Visible labels | Recall | Miss rate | Eligible pairs | Dropouts | False det/frame | ms/frame |
|---|---|---|---|---|---|---|---|---|
| indoor_person01 | dark | 60 | 0.967 | 0.033 | 59 | 1 | 0.03 | 607 |
| indoor_person01 | gamma_1.5 | 60 | 0.967 | 0.033 | 59 | 1 | 0.03 | 659 |
| indoor_person01 | gamma_2.5 | 60 | 0.967 | 0.033 | 59 | 1 | 0.03 | 602 |
| indoor_person01 | clahe | 60 | 0.967 | 0.033 | 59 | 1 | 0.03 | 688 |
| indoor_person01 | normal | 60 | 0.967 | 0.033 | 59 | 1 | 0.03 | 588 |
| outdoor_person04 | dark | 56 | 0.946 | 0.054 | 55 | 2 | 0.05 | 517 |
| outdoor_person04 | gamma_1.5 | 56 | 0.946 | 0.054 | 55 | 2 | 0.05 | 515 |
| outdoor_person04 | gamma_2.5 | 56 | 0.946 | 0.054 | 55 | 2 | 0.05 | 524 |
| outdoor_person04 | clahe | 56 | 0.946 | 0.054 | 55 | 2 | 0.05 | 531 |
| outdoor_person04 | normal | 56 | 0.946 | 0.054 | 55 | 2 | 0.05 | 514 |
| outdoor_person05 | dark | 60 | 0.983 | 0.017 | 59 | 1 | 0.03 | 481 |
| outdoor_person05 | gamma_1.5 | 60 | 1.000 | 0.000 | 59 | 0 | 0.02 | 476 |
| outdoor_person05 | gamma_2.5 | 60 | 0.983 | 0.017 | 59 | 1 | 0.02 | 479 |
| outdoor_person05 | clahe | 60 | 0.983 | 0.017 | 59 | 1 | 0.02 | 496 |
| outdoor_person05 | normal | 60 | 0.983 | 0.017 | 59 | 1 | 0.02 | 485 |
| outdoor_person02 | dark | 60 | 1.000 | 0.000 | 59 | 0 | 0.00 | 576 |
| outdoor_person02 | gamma_1.5 | 60 | 1.000 | 0.000 | 59 | 0 | 0.00 | 571 |
| outdoor_person02 | gamma_2.5 | 60 | 1.000 | 0.000 | 59 | 0 | 0.00 | 567 |
| outdoor_person02 | clahe | 60 | 1.000 | 0.000 | 59 | 0 | 0.00 | 549 |
| outdoor_person02 | normal | 60 | 1.000 | 0.000 | 59 | 0 | 0.00 | 546 |
| drawing | dark | 60 | 1.000 | 0.000 | 59 | 0 | 0.97 | 366 |
| drawing | gamma_1.5 | 60 | 1.000 | 0.000 | 59 | 0 | 0.00 | 362 |
| drawing | gamma_2.5 | 60 | 1.000 | 0.000 | 59 | 0 | 0.00 | 360 |
| drawing | clahe | 60 | 1.000 | 0.000 | 59 | 0 | 0.00 | 397 |
| drawing | normal | 60 | 1.000 | 0.000 | 59 | 0 | 0.00 | 376 |
| hand_gesture02 | dark | 60 | 1.000 | 0.000 | 59 | 0 | 0.00 | 541 |
| hand_gesture02 | gamma_1.5 | 60 | 1.000 | 0.000 | 59 | 0 | 0.00 | 539 |
| hand_gesture02 | gamma_2.5 | 60 | 1.000 | 0.000 | 59 | 0 | 0.00 | 539 |
| hand_gesture02 | clahe | 60 | 1.000 | 0.000 | 59 | 0 | 0.00 | 551 |
| hand_gesture02 | normal | 60 | 1.000 | 0.000 | 59 | 0 | 0.00 | 540 |
| whiteboard | dark | 60 | 0.767 | 0.233 | 59 | 3 | 0.20 | 553 |
| whiteboard | gamma_1.5 | 60 | 1.000 | 0.000 | 59 | 0 | 0.00 | 547 |
| whiteboard | gamma_2.5 | 60 | 0.967 | 0.033 | 59 | 2 | 0.00 | 541 |
| whiteboard | clahe | 60 | 0.983 | 0.017 | 59 | 1 | 0.08 | 554 |
| whiteboard | normal | 60 | 1.000 | 0.000 | 59 | 0 | 0.00 | 546 |
