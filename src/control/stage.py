from enum import IntEnum


class Stage(IntEnum):
    """Named stages used to pick one object and place it in its bin.

    The active controller normally follows ``RAISE`` through ``RELEASE`` in
    order. ``REALIGN_XY`` remains available for compatibility with earlier
    experiments but is not part of the current sequence.
    """

    # Numeric values are retained for compatibility with existing stage logs.
    RAISE = -1
    ALIGN_XY = 0
    DESCEND = 1
    ORIENT_YAW = 2
    REALIGN_XY = 4
    FINE_DESCEND = 5
    GRASP = 6
    LIFT = 7
    REZERO_YAW = 8
    MOVE_TO_BIN = 9
    DESCEND_TO_BIN = 10
    RELEASE = 11
