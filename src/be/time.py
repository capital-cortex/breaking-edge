import re as _re
import sys as _sys
import time as _time
import datetime as _datetime
import typing as _typing

from .ansi import Rainbow

__all__ = ["PlatformTime", "Timestamper"]

# TODO: implement LinuxTime()
# TODO: fix spaces

PlatformTime = None

if _sys.platform == "win32":
    import winreg as _winreg
    class WindowsTime():

        SYNC_INTERVAL_REG_SUBKEY = r"SYSTEM\CurrentControlSet\Services\W32Time\TimeProviders\NtpClient"
        SYNC_INTERVAL_REG_NAME   =  "SpecialPollInterval"
        SYNC_ITME_REG_SUBKEY     = r"SYSTEM\CurrentControlSet\Services\W32Time\Config"
        SYNC_TIME_REG_NAME       =  "LastKnownGoodTime"
        FILETIME_OFFSET_100NS    = 116444736000000000 # 1970-01-01 - 1601-01-01 in 100ns

        def __init__(self) -> None:
            self.last_sync_time_ns = self.get_sync_time_ns(tries=1, delay=0)
            if self.get_sync_interval_s() > 86400:
                print("Warning: Windows time synchronisation interval is greater than '24h'.")
                print("Consider setting it to a lower value like 0x1000 for better synchronisation (system reboot required).")
                print(f"Registry key: 'HKEY_LOCAL_MACHINE\\{self.SYNC_INTERVAL_REG_SUBKEY}\\{self.SYNC_INTERVAL_REG_NAME}'")
        
        def get_sync_interval_s(self) -> int:
            with _winreg.OpenKey(_winreg.HKEY_LOCAL_MACHINE, self.SYNC_INTERVAL_REG_SUBKEY) as key:
                value, _ = _winreg.QueryValueEx(key, self.SYNC_INTERVAL_REG_NAME)
                return value
        
        def get_sync_time_ns(self, tries: int = 45*2, delay: int = 30) -> int: # TODO: tries -> attempts
            attempt = 0
            sync_time_ns : int = -1
            for attempt in range(1, tries + 1):
                try:
                    with _winreg.OpenKey(_winreg.HKEY_LOCAL_MACHINE, self.SYNC_ITME_REG_SUBKEY) as key:
                        value, _ = _winreg.QueryValueEx(key, self.SYNC_TIME_REG_NAME)
                        sync_time_ns = (value - self.FILETIME_OFFSET_100NS) * 100
                        break
                except Exception as e:
                    print(f"An error occurred while querying windows time synchronisation on attempt {attempt}/{tries}:")
                    print(e)
                    if attempt < tries:
                        _time.sleep(delay)
                        continue
                    print("Raising Exception. Goodbye :(")
                    raise
            if attempt > 1:
                print(f"Succeeded querying windows time synchronisation on attempt {attempt}/{tries} :)")
            return sync_time_ns

        def was_synced(self, tries: int = 45*2, delay: int = 30) -> bool:
            sync_time_ns = self.get_sync_time_ns(tries, delay)
            if sync_time_ns == self.last_sync_time_ns:
                return False
            self.last_sync_time_ns = sync_time_ns
            return True
    
    PlatformTime = WindowsTime
if _sys.platform == "linux":
    class LinuxTime():
        def __init__(self) -> None:
            raise NotImplementedError()
        def was_synced(self) -> bool:
            raise NotImplementedError()
    PlatformTime = LinuxTime

class Timestamper():

    ANSI_ESCAPE = _re.compile(r'\x1b\[[0-9;]*[mGKH]')

    def __init__(
            self                               ,
            format   : str        = "%F %T.%3f",
            log_path : str | None = None       ,
    ) -> None:
        self.stdout   = _sys.stdout
        self.updated  = False
        self.buffer   = ""
        self.rainbow  = Rainbow()
        self.format   = format
        self.log_path = log_path
        _sys.stdout = self
    
    def __getattr__(self, name: str) -> _typing.Any:
        return getattr(self.stdout, name)
    
    @staticmethod
    def format_timestamp(
            timestamp : _datetime.datetime | None = None       ,
            format    : str                       = "%F %T.%3f",
    ) -> str:
        timestamp = timestamp or _datetime.datetime.now(_datetime.timezone.utc)
        format = _re.sub(
            pattern = r"%([1-6])?f",
            repl    = lambda x: f"{timestamp.microsecond:06d}"[:int(x.group(1)) if x.group(1) else 6],
            string  = format
        )
        return f"{timestamp:{format}}"
    
    def flush(self) -> None:
        self.stdout.flush()
    
    def write(self, text: str) -> None:
        self.buffer += text
        if not "\n" in self.buffer:
            return
        *lines, self.buffer = self.buffer.split("\n")
        if self.log_path is None:
            if self.updated:
                self.stdout.write("\n")
                self.updated = False
            for line in lines:
                self.stdout.write(line + "\n")
                self.flush()
            return
        try:
            with open(self.log_path, "a") as f:
                if self.updated:
                    self.stdout.write("\n")
                    f.write("\n" + self.format_timestamp(format=self.format) + "\n")
                    self.updated = False
                for line in lines:
                    self.stdout.write(line + "\n")
                    f.write(self.ANSI_ESCAPE.sub("", line + "\n"))
                    self.flush()
        except Exception as e:
            print(f"Failed to log the print to '{self.log_path}':")
            print(e)
    
    def writelines(self, lines: list[str]) -> None:
        for line in lines:
            self.write(line)
    
    def update(self, timestamp: _datetime.datetime | None = None) -> None:
        if not self.updated:
            self.stdout.write("\n")
        self.stdout.write("\r" + self.rainbow.paint_delimited(self.format_timestamp(timestamp, self.format)))
        self.flush()
        self.updated = True
    
    def restore(self) -> None:
        _sys.stdout = self.stdout
