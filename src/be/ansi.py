import re as _re
import typing as _typing

class Rainbow():
    
    BLACK         = (0x00, 0x00, 0x00)
    RED           = (0xFF, 0x00, 0x00)
    YELLOW        = (0xFF, 0xFF, 0x00)
    GREEN         = (0x00, 0xFF, 0x00)
    CYAN          = (0x00, 0xFF, 0xFF)
    BLUE          = (0x00, 0x00, 0xFF)
    MAGENTA       = (0xFF, 0x00, 0xFF)
    LIGHT_BLACK   = (0x7F, 0x7F, 0x7F)
    LIGHT_RED     = (0xFF, 0x7F, 0x7F)
    LIGHT_YELLOW  = (0xFF, 0xFF, 0x7F)
    LIGHT_GREEN   = (0x7F, 0xFF, 0x7F)
    LIGHT_CYAN    = (0x7F, 0xFF, 0xFF)
    LIGHT_BLUE    = (0x7F, 0x7F, 0xFF)
    LIGHT_MAGENTA = (0xFF, 0x7F, 0xFF)
    WHITE         = (0xFF, 0xFF, 0xFF)
    
    RGB_F = "\x1b[38;2;{r};{g};{b}m{text}\x1b[0m"
    
    def __init__(
            self,
            glow  : int        = 0x7F,
            bpc   : int        =   20,
            step  : int | None = None,
            freq  : int        =    1,
            state : int | None = None,
        ) -> None:
        assert 0 <= glow <= 0xFF, "Invalid arg value: 'glow' must be an integer between 0 and 255."
        assert bpc > 0          , "Invalid arg value: 'bpc' must be a positive integer."
        self.glow  = glow
        self.bpc   = bpc     # bits per color
        self.bpp   = bpc * 3 # bits per pixel
        self.mask  = (1 << self.bpp) - 1
        self.masks = [(1 << self.bpc) - 1 << self.bpc * i for i in reversed(range(3))]
        self.step  = step if step is not None else bpc // 2
        self.freq  = freq
        self.state = (
            state if state is not None else
            ((1 << self.bpp // 2) - 1 << self.bpp // 4) ^ self.mask
        ) & self.mask
        assert isinstance(self.glow , int), "Invalid arg type: 'glow' must be of type int."
        assert isinstance(self.bpc  , int), "Invalid arg type: 'bpc' must be of type int."
        assert isinstance(self.step , int), "Invalid arg type: 'step' must be of type int | None."
        assert isinstance(self.freq , int), "Invalid arg type: 'freq' must be of type int."
        assert isinstance(self.state, int), "Invalid arg type: 'state' must be of type int | None."
    
    @staticmethod
    def get_color(val: int | float) -> tuple[int, int, int]:
        return Rainbow.LIGHT_GREEN if val > 0 else Rainbow.LIGHT_RED if val < 0 else Rainbow.LIGHT_CYAN
    
    @staticmethod
    def paint(text: str, color: tuple[int, int, int]) -> str:
        r, g, b = color
        return Rainbow.RGB_F.format(text=text, r=r, g=g, b=b)
    
    @staticmethod
    def paint_value(val: int | float | str) -> str:
        try:
            numeric_val = float(val)
        except:
            return str(val)
        return Rainbow.paint(str(val), Rainbow.get_color(numeric_val))
    
    def rotr(self, val: int, shift: int) -> int:
        shift %= self.bpp
        return ((val >> shift) | (val << (self.bpp - shift))) & self.mask
    
    def state_to_rgb(self, state: int) -> tuple[int, int, int]:
        rgbs = [(state & self.masks[i]).bit_count() for i in range(3)]
        rgbs = [self.glow + ((0xFF - self.glow) * c // self.bpc) for c in rgbs]
        return _typing.cast(tuple[int, int, int], tuple(rgbs))
    
    def paint_delimited(self, text: str, re_delimiters: str = "[^a-zA-Z0-9]") -> str:
        if not text.strip():
            return text
        lines = text.split("\n")
        if len(lines) > 1:
            return "\n".join(self.paint_delimited(line, re_delimiters) for line in lines)
        chunk_state = self.state
        def render_chunk(text: str) -> str:
            nonlocal chunk_state
            r, g, b = self.state_to_rgb(chunk_state)
            chunk_state = self.rotr(chunk_state, self.step)
            return self.RGB_F.format(text=text, r=r, g=g, b=b)
        stripped_text = text.strip(" ")
        delim_pattern = _re.compile(f"^{re_delimiters}+$")
        tokens: list[str] = _re.split(f"({re_delimiters}+)", stripped_text)
        painted = "".join(
            self.paint(t, self.WHITE) if delim_pattern.match(t) else render_chunk(t)
            for t in tokens if t
        )
        self.state = self.rotr(self.state, self.freq)
        lspace = " " * (len(text) - len(text.lstrip(" ")))
        rspace = " " * (len(text) - len(text.rstrip(" ")))
        return f"{lspace}{painted}{rspace}"
