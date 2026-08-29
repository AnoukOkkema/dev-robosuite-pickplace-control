import numpy as np
from robosuite.environments.manipulation.pick_place import PickPlace


class PickPlaceWithRobotOffset(PickPlace):
    """Apply a world-frame robot offset before MuJoCo compiles the model.

    This small environment subclass leaves Robosuite's PickPlace task intact.
    It just makes the robot position configurable, for reachability experiments.
    """

    def __init__(self, *args, robot_base_offset=(0.0, 0.0, 0.0), **kwargs):
        """Initialize the environment with an optional XYZ base offset.

        Args:
            *args: Positional arguments accepted by :class:`PickPlace`.
            robot_base_offset: World-frame XYZ offset in metres.
            **kwargs: Keyword arguments accepted by :class:`PickPlace`.

        Raises:
            ValueError: If ``robot_base_offset`` does not contain three values.
        """
        self.robot_base_offset = np.asarray(robot_base_offset, dtype=float)
        if self.robot_base_offset.shape != (3,):
            raise ValueError("robot_base_offset must have three coordinates")
        super().__init__(*args, **kwargs)

    def _load_model(self):
        """Load the model and apply the offset before compilation."""
        super()._load_model()
        if np.any(self.robot_base_offset):
            # ``bins`` is the task's default Panda-base reference position.
            base_position = self.robots[0].robot_model.base_xpos_offset["bins"]
            self.robots[0].robot_model.set_base_xpos(
                np.asarray(base_position) + self.robot_base_offset
            )
