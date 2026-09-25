import os as _os
import re as _re
import sys as _sys
import time as _time
import datetime as _datetime
import subprocess as _subprocess
import typing as _typing

from .ansi import Rainbow

# TODO: use tenacity

__all__ = ["PlatformTime", "PlatformTimeProtocol", "Timestamper"]

class PlatformTimeProtocol(_typing.Protocol):
    def was_synced(self, tries: int = 45*2, delay: int = 30) -> bool: ...

class _UnsupportedPlatformTime(PlatformTimeProtocol):
    def __init__(self) -> None:
        raise NotImplementedError(f"No 'PlatformTime' implementation for '{_sys.platform}'.")

PlatformTime : type[PlatformTimeProtocol] = _UnsupportedPlatformTime

if _sys.platform == "win32":
    import winreg as _winreg
    class WindowsTime(PlatformTimeProtocol):
        
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
    class LinuxTime(PlatformTimeProtocol):
        
        STEP_TOLERANCE_NS  = 1_000_000_000 # 1s, well above the drift of a slewing clock
        SYSTEMD_SYNC_PATHS = ("/run/systemd/timesync/synchronized", "/var/lib/systemd/timesync/clock")
        CHRONY_CMD         = ("chronyc", "tracking")
        CHRONY_REF_TIME_F  = r"Ref time \(UTC\)\s*:\s*(.+)"
        CHRONY_UNSYNCED_F  = r"unspecified|not synchronis"
        CHRONY_TIME_OUT_S  = 5
        
        def __init__(self) -> None:
            self.last_sync_time_ns   = self.get_sync_time_ns(tries=1, delay=0)
            self.last_clock_delta_ns = self.get_clock_delta_ns()
            if self.last_sync_time_ns == 0:
                print("Warning: No Linux time synchronisation source found (e.g. 'systemd-timesyncd' or 'chronyd').")
                print("Consider running an NTP client (e.g. 'sudo timedatectl set-ntp true') to have the clock corrected regularly.")
        
        @staticmethod
        def get_clock_delta_ns() -> int:
            # 'CLOCK_BOOTTIME' keeps running while suspended, so suspending is no clock step
            return _time.clock_gettime_ns(_time.CLOCK_REALTIME) - _time.clock_gettime_ns(_time.CLOCK_BOOTTIME)
        
        @staticmethod
        def get_systemd_sync_time_ns() -> int:
            mtime_ns_list : list[int] = []
            for path in LinuxTime.SYSTEMD_SYNC_PATHS:
                try:
                    mtime_ns_list.append(_os.stat(path).st_mtime_ns)
                except OSError:
                    continue
            return max(mtime_ns_list, default=0)
        
        @staticmethod
        def get_chrony_sync_time_ns() -> int:
            try:
                tracking = _subprocess.run(
                    LinuxTime.CHRONY_CMD,
                    capture_output = True,
                    text           = True,
                    timeout        = LinuxTime.CHRONY_TIME_OUT_S,
                ).stdout
            except (OSError, _subprocess.SubprocessError):
                return 0
            if _re.search(LinuxTime.CHRONY_UNSYNCED_F, tracking, _re.IGNORECASE):
                return 0
            match = _re.search(LinuxTime.CHRONY_REF_TIME_F, tracking)
            if match is None:
                return 0
            try:
                ref_time = _datetime.datetime.strptime(match.group(1).strip(), "%a %b %d %H:%M:%S %Y").replace(tzinfo=_datetime.timezone.utc)
            except ValueError:
                return 0
            return int(ref_time.timestamp() * 1e9)
        
        def get_sync_time_ns(self, tries: int = 45*2, delay: int = 30) -> int:
            attempt = 0
            sync_time_ns : int = 0
            for attempt in range(1, tries + 1):
                try:
                    sync_time_ns = max(self.get_systemd_sync_time_ns(), self.get_chrony_sync_time_ns())
                    break
                except Exception as e:
                    print(f"An error occurred while querying linux time synchronisation on attempt {attempt}/{tries}:")
                    print(e)
                    if attempt < tries:
                        _time.sleep(delay)
                        continue
                    print("Raising Exception. Goodbye :(")
                    raise
            if attempt > 1:
                print(f"Succeeded querying linux time synchronisation on attempt {attempt}/{tries} :)")
            return sync_time_ns
        
        def was_synced(self, tries: int = 45*2, delay: int = 30) -> bool:
            clock_delta_ns = self.get_clock_delta_ns()
            was_stepped    = abs(clock_delta_ns - self.last_clock_delta_ns) > self.STEP_TOLERANCE_NS
            sync_time_ns   = self.get_sync_time_ns(tries, delay)
            if not was_stepped and sync_time_ns == self.last_sync_time_ns:
                return False
            self.last_sync_time_ns   = sync_time_ns
            self.last_clock_delta_ns = clock_delta_ns
            return True
    
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
