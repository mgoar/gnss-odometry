# A GNSS odometry implementation using time differences of carrier phase (TDCP) measurements and factor graph optimization
Quick-and-dirty implementation based on:
> T. Suzuki, "GNSS Odometry: Precise Trajectory Estimation Based on Carrier Phase Cycle Slip Estimation," in IEEE Robotics and Automation Letters, vol. 7, no. 3, pp. 7319-7326, July 2022, doi: 10.1109/LRA.2022.3182795

and using [GTSAM](https://gtsam.org).

Vanilla implementation with some modifications:
- GPS only
- No loop closure (60 s)
- No loop lock indicator (LLI) flag
- Evaluation using Vincenty's formula instead of ATE (absolute trajectory error) + score (comparison against Weighted Least Squares, WLS)