KEY_UP = {ord("W"), ord("w"), 265}
KEY_DOWN = {ord("S"), ord("s"), 264}
KEY_LEFT = {ord("A"), ord("a"), 263}
KEY_RIGHT = {ord("D"), ord("d"), 262}
KEY_STOP = {32, 256}


class ViewerCommand:
    def __init__(self):
        self.motion_command = "idle"

    def handle_key(self, keycode):
        if keycode in KEY_UP:
            self.motion_command = "forward"
        elif keycode in KEY_DOWN:
            self.motion_command = "backward"
        elif keycode in KEY_LEFT:
            self.motion_command = "turn_left"
        elif keycode in KEY_RIGHT:
            self.motion_command = "turn_right"
        elif keycode in KEY_STOP:
            self.motion_command = "idle"

    @property
    def key_up(self):
        return self.motion_command == "forward"

    @property
    def key_down(self):
        return self.motion_command == "backward"

    @property
    def key_left(self):
        return self.motion_command == "turn_left"

    @property
    def key_right(self):
        return self.motion_command == "turn_right"

