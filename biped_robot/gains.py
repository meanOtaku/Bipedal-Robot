KP = {"yaw": 1000, "roll": 1500, "hip": 1200, "knee": 1500, "ankle_pitch": 1000, "ankle_roll": 1000}
KD = {"yaw": 60, "roll": 250, "hip": 150, "knee": 180, "ankle_pitch": 150, "ankle_roll": 150}


def get_gains(name, walk_mode, state):
    if "yaw" in name:
        return 50.0, 5.0
    if "hip_roll" in name:
        return KP["roll"], KD["roll"]
    if "hip_pitch" in name:
        return KP["hip"], KD["hip"]
    if "knee" in name:
        return KP["knee"], KD["knee"]
    if "ankle_pitch" in name:
        if walk_mode == "forward":
            if "left" in name and state in [50, 51, 6]:
                return 100, 10
            if "right" in name and state in [80, 81, 3]:
                return 100, 10
        return KP["ankle_pitch"], KD["ankle_pitch"]
    if "ankle_roll" in name:
        return KP["ankle_roll"], KD["ankle_roll"]
    return 100, 10

