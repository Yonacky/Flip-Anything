"""Process-lifetime Windows leases for cooperating hardware entrypoints."""
import ctypes,hashlib,os
from contextlib import ExitStack,contextmanager
from ctypes import wintypes

class DeviceBusy(RuntimeError):pass

@contextmanager
def named_lease(identity):
    if os.name!='nt':raise RuntimeError('This hardware workspace requires Windows')
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateMutexW.argtypes=[wintypes.LPVOID,wintypes.BOOL,wintypes.LPCWSTR]
    kernel.CreateMutexW.restype=wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD]
    kernel.WaitForSingleObject.restype=wintypes.DWORD
    kernel.ReleaseMutex.argtypes=[wintypes.HANDLE];kernel.ReleaseMutex.restype=wintypes.BOOL
    kernel.CloseHandle.argtypes=[wintypes.HANDLE];kernel.CloseHandle.restype=wintypes.BOOL
    name='Global\\FlipAnything.'+hashlib.sha256(identity.encode()).hexdigest()
    handle=kernel.CreateMutexW(None,False,name)
    if not handle:raise ctypes.WinError(ctypes.get_last_error())
    owned=False
    try:
        result=kernel.WaitForSingleObject(handle,0)
        if result==258:raise DeviceBusy('Another cooperating session owns '+identity)
        if result not in (0,128):raise ctypes.WinError(ctypes.get_last_error())
        owned=True
        yield
    finally:
        if owned:kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)

@contextmanager
def device_lease(config):
    with ExitStack() as stack:
        # Separate identities prevent same arm/different-camera configurations racing.
        stack.enter_context(named_lease('arm:'+config['robot']['ip']))
        stack.enter_context(named_lease('camera:'+config['camera']['serial']))
        yield
