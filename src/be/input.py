import os
import sys
import time
import atexit
import types
from typing import List

if os.name == "nt":
    import msvcrt
else:
    import codecs
    import select
    import termios

# CSI <final> and SS3 <final> sequences: ESC [ A, ESC O P, ...
_CSI_FINAL = {
    "A": "up", "B": "down", "C": "right", "D": "left",
    "E": "begin", "F": "end", "H": "home",
    "P": "f1", "Q": "f2", "R": "f3", "S": "f4",
}
# CSI <number> ~ sequences: ESC [ 15 ~ == F5, ...
_CSI_TILDE = {
    "1": "home", "2": "insert", "3": "delete", "4": "end",
    "5": "page_up", "6": "page_down", "7": "home", "8": "end",
    "11": "f1", "12": "f2", "13": "f3", "14": "f4", "15": "f5",
    "17": "f6", "18": "f7", "19": "f8", "20": "f9", "21": "f10",
    "23": "f11", "24": "f12", "25": "f13", "26": "f14",
    "28": "f15", "29": "f16", "31": "f17", "32": "f18",
    "33": "f19", "34": "f20",
}
# Linux virtual console: ESC [ [ A .. E == F1 .. F5
_LINUX_CONSOLE_F = {"A": "f1", "B": "f2", "C": "f3", "D": "f4", "E": "f5"}
# Windows: second code returned by msvcrt.getwch() after a "\x00" / "\xe0" prefix
_WIN_SPECIAL = {
    "H": "up", "P": "down", "K": "left", "M": "right",
    "G": "home", "O": "end", "I": "page_up", "Q": "page_down",
    "R": "insert", "S": "delete",
}
for _n in range(10):  # F1-F10 (plain, shift, ctrl, alt)
    _WIN_SPECIAL[chr(0x3B + _n)] = f"f{_n + 1}"
    _WIN_SPECIAL[chr(0x54 + _n)] = f"shift+f{_n + 1}"
    _WIN_SPECIAL[chr(0x5E + _n)] = f"ctrl+f{_n + 1}"
    _WIN_SPECIAL[chr(0x68 + _n)] = f"alt+f{_n + 1}"
for _mod, _f11, _f12 in (("", 0x85, 0x86), ("shift+", 0x87, 0x88),
                         ("ctrl+", 0x89, 0x8A), ("alt+", 0x8B, 0x8C)):
    _WIN_SPECIAL[chr(_f11)] = _mod + "f11"
    _WIN_SPECIAL[chr(_f12)] = _mod + "f12"

def _unknown(raw: str) -> str:
    return "unknown:" + raw.encode("unicode_escape").decode("ascii")

def _decode_char(ch: str) -> str:
    """Name a single (non-escape-sequence) character."""
    if ch in ("\r", "\n"):
        return "enter"
    if ch == "\t":
        return "tab"
    if ch in ("\x7f", "\x08"):
        return "backspace"
    if ch == "\x1b":
        return "esc"
    if ch == "\x00":
        return "ctrl+space"
    o = ord(ch)
    if 1 <= o <= 26:
        return "ctrl+" + chr(o + 96)
    if 28 <= o <= 31:
        return "ctrl+" + chr(o + 64)
    return ch

def _csi_key(params: str, final: str, raw: str) -> str:
    """Translate a complete CSI sequence (ESC [ params final) into a key name."""
    parts = params.split(";")
    # xterm style modifier: 2=shift 3=alt 4=shift+alt 5=ctrl 6=ctrl+shift ...
    mod = int(parts[1]) - 1 if len(parts) > 1 and parts[1].isdigit() else 0
    if final == "~":
        name = _CSI_TILDE.get(parts[0])
    elif final == "Z":
        name, mod = "tab", mod | 1  # Shift+Tab
    else:
        name = _CSI_FINAL.get(final)
    if name is None:
        return _unknown(raw)
    prefix = ""
    if mod & 4:
        prefix += "ctrl+"
    if mod & 2:
        prefix += "alt+"
    if mod & 1:
        prefix += "shift+"
    if mod & 8:
        prefix += "meta+"
    return prefix + name

def _parse_escape(buf: str) -> None | tuple[str, int]:
    """
    buf starts with ESC. Return (key_name, chars_consumed), or None if buf is
    only the beginning of a longer sequence and more input is needed.
    """
    if len(buf) < 2:
        return None
    c = buf[1]
    if c == "[":
        if len(buf) < 3:
            return None
        if buf[2] == "[":  # Linux console F1-F5
            if len(buf) < 4:
                return None
            return (_LINUX_CONSOLE_F.get(buf[3]) or _unknown(buf[:4])), 4
        i = 2
        while i < len(buf) and "\x20" <= buf[i] <= "\x3f":  # params/intermediates
            i += 1
        if i >= len(buf):
            return None
        if not "\x40" <= buf[i] <= "\x7e":  # malformed: give up on the sequence
            return "esc", 1
        return _csi_key(buf[2:i], buf[i], buf[:i + 1]), i + 1
    if c == "O":  # SS3: F1-F4 and application-mode arrows
        if len(buf) < 3:
            return None
        return (_CSI_FINAL.get(buf[2]) or _unknown(buf[:3])), 3
    if c == "\x1b":
        return "esc", 1
    return "alt+" + _decode_char(c), 2  # ESC <char> == Alt+<char>

class KeyReader:
    """
    Non-blocking keyboard reader. Use as a context manager (recommended) or
    call start() / stop() yourself.

    esc_timeout: seconds to wait after a lone ESC byte before deciding it is
    the Escape key rather than the start of an escape sequence (Unix only).
    """
    def __init__(self, esc_timeout: float = 0.05) -> None:
        self._esc_timeout = esc_timeout
        self._active = False
        # Unix state
        self._fd = -1
        self._old_attrs = None
        self._decoder = None
        self._buf = ""
        self._last_rx = 0.0

    def __enter__(self) -> "KeyReader":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException]  | None,
        exc_value: BaseException       | None,
        traceback: types.TracebackType | None,
    ) -> None:
        self.stop()

    def start(self) -> None:
        if self._active:
            return
        if sys.stdin is None or not sys.stdin.isatty():
            raise RuntimeError("KeyReader needs an interactive terminal on stdin")
        if not os.name == "nt":
            self._fd = sys.stdin.fileno()
            self._old_attrs = termios.tcgetattr(self._fd)
            attrs = termios.tcgetattr(self._fd)
            # cbreak: no line buffering, no echo, but Ctrl+C etc. still work
            attrs[3] &= ~(termios.ICANON | termios.ECHO)
            attrs[6][termios.VMIN] = 1
            attrs[6][termios.VTIME] = 0
            termios.tcsetattr(self._fd, termios.TCSADRAIN, attrs)
            self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
            self._buf = ""
        self._active = True
        atexit.register(self.stop)  # restore the terminal even on a crash

    def stop(self) -> None:
        if not self._active:
            return
        self._active = False
        atexit.unregister(self.stop)
        if not os.name == "nt" and self._old_attrs is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)
            except (termios.error, OSError):
                pass
            self._old_attrs = None

    def read_key(self) -> str | None:
        """Return the next key as a string, or None if no key is waiting."""
        if not self._active:
            raise RuntimeError("KeyReader is not started; use 'with KeyReader():'")
        if os.name == "nt":
            if not msvcrt.kbhit():
                return None
            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):  # prefix for function / cursor keys
                code = msvcrt.getwch()
                return _WIN_SPECIAL.get(code) or _unknown(ch + code)
            if "\ud800" <= ch <= "\udbff" and msvcrt.kbhit():  # surrogate pair
                low = msvcrt.getwch()
                ch = (ch + low).encode("utf-16", "surrogatepass").decode("utf-16", "replace")
            return _decode_char(ch)
        else:
            while True:
                ready, _, _ = select.select([self._fd], [], [], 0)
                if not ready:
                    break
                data = os.read(self._fd, 4096)
                if not data:  # EOF
                    break
                self._buf += self._decoder.decode(data)
                self._last_rx = time.monotonic()
            buf = self._buf
            if not buf:
                return None
            if buf[0] != "\x1b":
                self._buf = buf[1:]
                return _decode_char(buf[0])
            parsed = _parse_escape(buf)
            if parsed is not None:
                key, used = parsed
                self._buf = buf[used:]
                return key
            # Incomplete escape sequence: wait briefly for the rest to arrive.
            if time.monotonic() - self._last_rx < self._esc_timeout:
                return None
            if len(buf) == 2:  # "ESC [" or "ESC O" typed as Alt+[ / Alt+O
                self._buf = buf[2:]
                return "alt+" + buf[1]
            self._buf = buf[1:]  # a lone ESC key press
            return "esc"

    def read_keys(self) -> List[str]:
        """Return all keys waiting right now (possibly an empty list)."""
        keys: list[str] = []
        while True:
            key = self.read_key()
            if key is None:
                return keys
            keys.append(key)
