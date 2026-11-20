# Scores (PekingU/rtdetr_r50vd, person, threshold 0.5, IoU 0.5)

Recall = matched visible person boxes / visible person boxes. Dropouts = person matched in one frame and missed in the next (eligible = visible in both frames, same track ID). Miss rate is shown next to dropouts because the arms have different denominators.

| Clip | Arm | Visible labels | Recall | Miss rate | Eligible pairs | Dropouts | False det/frame | ms/frame |
|---|---|---|---|---|---|---|---|---|
| indoor_person01 | dark | 60 | 0.967 | 0.033 | 59 | 1 | 0.03 | 405 |
| indoor_person01 | gamma_1.5 | 60 | 0.967 | 0.033 | 59 | 1 | 0.03 | 384 |
| indoor_person01 | gamma_2.5 | 60 | 0.967 | 0.033 | 59 | 1 | 0.03 | 380 |
| indoor_person01 | clahe | 60 | 0.967 | 0.033 | 59 | 1 | 0.03 | 387 |
| indoor_person01 | normal | 60 | 0.967 | 0.033 | 59 | 1 | 0.03 | 389 |
| outdoor_person04 | dark | 56 | 0.946 | 0.054 | 55 | 2 | 0.05 | 375 |
| outdoor_person04 | gamma_1.5 | 56 | 0.946 | 0.054 | 55 | 2 | 0.05 | 372 |
| outdoor_person04 | gamma_2.5 | 56 | 0.946 | 0.054 | 55 | 2 | 0.05 | 365 |
| outdoor_person04 | clahe | 56 | 0.946 | 0.054 | 55 | 2 | 0.05 | 389 |
| outdoor_person04 | normal | 56 | 0.946 | 0.054 | 55 | 2 | 0.05 | 369 |
| outdoor_person05 | dark | 60 | 0.983 | 0.017 | 59 | 1 | 0.02 | 367 |
| outdoor_person05 | gamma_1.5 | 60 | 0.983 | 0.017 | 59 | 1 | 0.02 | 368 |
| outdoor_person05 | gamma_2.5 | 60 | 0.983 | 0.017 | 59 | 1 | 0.02 | 372 |
| outdoor_person05 | clahe | 60 | 0.983 | 0.017 | 59 | 1 | 0.02 | 382 |
| outdoor_person05 | normal | 60 | 0.983 | 0.017 | 59 | 1 | 0.02 | 366 |
| outdoor_person02 | dark | 60 | 1.000 | 0.000 | 59 | 0 | 0.00 | 376 |
| outdoor_person02 | gamma_1.5 | 60 | 1.000 | 0.000 | 59 | 0 | 0.00 | 406 |
| outdoor_person02 | gamma_2.5 | 60 | 1.000 | 0.000 | 59 | 0 | 0.00 | 390 |
| outdoor_person02 | clahe | 60 | 1.000 | 0.000 | 59 | 0 | 0.00 | 405 |
| outdoor_person02 | normal | 60 | 1.000 | 0.000 | 59 | 0 | 0.00 | 423 |
| drawing | dark | 60 | 1.000 | 0.000 | 59 | 0 | 0.00 | 397 |
| drawing | gamma_1.5 | 60 | 1.000 | 0.000 | 59 | 0 | 0.00 | 422 |
| drawing | gamma_2.5 | 60 | 1.000 | 0.000 | 59 | 0 | 0.00 | 382 |
| drawing | clahe | 60 | 1.000 | 0.000 | 59 | 0 | 0.00 | 436 |
| drawing | normal | 60 | 1.000 | 0.000 | 59 | 0 | 0.00 | 394 |
| hand_gesture02 | dark | 60 | 1.000 | 0.000 | 59 | 0 | 0.00 | 388 |
| hand_gesture02 | gamma_1.5 | 60 | 1.000 | 0.000 | 59 | 0 | 0.00 | 451 |
| hand_gesture02 | gamma_2.5 | 60 | 1.000 | 0.000 | 59 | 0 | 0.00 | 378 |
| hand_gesture02 | clahe | 60 | 1.000 | 0.000 | 59 | 0 | 0.00 | 386 |
| hand_gesture02 | normal | 60 | 1.000 | 0.000 | 59 | 0 | 0.00 | 377 |
| whiteboard | dark | 60 | 1.000 | 0.000 | 59 | 0 | 0.00 | 427 |
| whiteboard | gamma_1.5 | 60 | 1.000 | 0.000 | 59 | 0 | 0.00 | 370 |
| whiteboard | gamma_2.5 | 60 | 1.000 | 0.000 | 59 | 0 | 0.00 | 369 |
| whiteboard | clahe | 60 | 1.000 | 0.000 | 59 | 0 | 0.00 | 376 |
| whiteboard | normal | 60 | 1.000 | 0.000 | 59 | 0 | 0.00 | 381 |
