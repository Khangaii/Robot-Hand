"""D3-G mode input via Ubuntu 22.04's libgpiod v1 and an external pull-up."""
import ctypes as c
from ctypes.util import find_library
import os


class ModeSwitch:
    def __init__(self, device, offset):
        library = find_library("gpiod")
        if not library:
            raise RuntimeError("GPIO switch requires Ubuntu package libgpiod2.")
        self.lib = c.CDLL(library, use_errno=True)
        ptr, string, integer = c.c_void_p, c.c_char_p, c.c_int
        functions = [
            ("gpiod_chip_open", ptr, [string]),
            ("gpiod_chip_get_line", ptr, [ptr, c.c_uint]),
            ("gpiod_line_request_input", integer, [ptr, string]),
            ("gpiod_line_get_value", integer, [ptr]),
            ("gpiod_chip_close", None, [ptr]),
        ]
        for name, result, arguments in functions:
            method = getattr(self.lib, name)
            method.restype = result
            method.argtypes = arguments
        self.chip = self.lib.gpiod_chip_open(os.fsencode(device))
        if not self.chip:
            raise OSError(c.get_errno(), os.strerror(c.get_errno()), device)
        try:
            self.line = self.lib.gpiod_chip_get_line(self.chip, offset)
            if not self.line or self.lib.gpiod_line_request_input(self.line, b"HUANG-mode") < 0:
                raise OSError(c.get_errno(), os.strerror(c.get_errno()), f"{device} line {offset}")
        except Exception:
            self.close()
            raise
        self.pending = self.accepted = None
        self.since = 0.0

    def change(self, now):
        level = self.lib.gpiod_line_get_value(self.line)
        if level < 0:
            raise OSError(c.get_errno(), os.strerror(c.get_errno()))
        if level != self.pending:
            self.pending, self.since = level, now
        if level != self.accepted and now - self.since >= 0.05:
            self.accepted = level
            return "follow" if level else "game"
        return None

    def close(self):
        if self.chip:
            self.lib.gpiod_chip_close(self.chip)
            self.chip = None
