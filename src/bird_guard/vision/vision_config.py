from dataclasses import dataclass, field
from typing import Any



# ========================
# SubSubConfig ActivityMap
# ========================
@dataclass
class SubSubConfig_ActivityMap:
    """
    Settings for the activity map
    """
    cell_size: tuple[int, int] = (10, 10)       # cell size
    alpha: float = 0.02                         # alpha parameter of the cv2.accumulateWeighted function

    @classmethod
    def from_dict(cls, config_file_data_vision: dict[str, Any]) -> "SubSubConfig_ActivityMap":
        return cls(
            cell_size=tuple(config_file_data_vision.get("cell_size", cls.cell_size)),
            alpha=float(config_file_data_vision.get("alpha", cls.alpha)),
        )


# ========================
# SubConfig MotionDetector
# ========================
@dataclass
class SubConfig_MotionDetector:
    """
    Settings for the motion detector
    """
    activity_map: SubSubConfig_ActivityMap = field(default_factory=SubSubConfig_ActivityMap)

    @classmethod
    def from_dict(cls, config_file_data_vision: dict[str, Any]) -> "SubConfig_MotionDetector":
        return cls(
            activity_map=SubSubConfig_ActivityMap.from_dict(config_file_data_vision.get("activity_map", {})),
        )


# ===================
# ModuleConfig Camera
# ===================
@dataclass()
class ModuleConfig_Vision:
    """
    General vision settings
    """
    debug: bool = False                                        # enable debugging window and views
    motion_detector: SubConfig_MotionDetector = field(default_factory=SubConfig_MotionDetector)

    @classmethod
    def from_dict(cls, config_file_data_vision: dict[str, Any]) -> "ModuleConfig_Vision":
        return cls(
            motion_detector=SubConfig_MotionDetector.from_dict(config_file_data_vision.get("motion_detector", {})),
            debug=bool(config_file_data_vision.get("debug", cls.debug)),
        )