"""Global (system-wide) hotkey that brings the Jarvis window forward and
focuses its input box, from any application.

A web page can only see keystrokes while its own window has focus, so this
registers an OS-level hotkey with the Win32 RegisterHotKey API (no extra
dependency). When it fires it (a) tells the UI to focus the input box, via
state.request_focus() which the page notices on its next poll, and (b) raises
the browser window showing Jarvis, found by its title.

NOTE: a registered global hotkey is *swallowed* system-wide while Jarvis is
running -- Ctrl+Space in other apps (e.g. autocomplete in code editors, IME
switching) won't reach them. Change HOTKEY_MODIFIERS / HOTKEY_VK below if that
gets in the way.
"""

import ctypes
import ctypes.wintypes as wt
import threading
import time

MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_NOREPEAT = 0x4000

HOTKEY_MODIFIERS = MOD_CONTROL
HOTKEY_VK = 0x20            # VK_SPACE
HOTKEY_LABEL = "Ctrl+Space"

# Substring of the browser window title to raise. templates/index.html's
# <title> is "J.A.R.V.I.S."; browsers append their own name after it.
WINDOW_TITLE_MARKER = "J.A.R.V.I.S."

_WM_HOTKEY = 0x0312
_PM_REMOVE = 0x0001
_SW_RESTORE = 9
_HOTKEY_ID = 0x4A52   # arbitrary, unique within this thread

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

user32.RegisterHotKey.argtypes = [wt.HWND, ctypes.c_int, wt.UINT, wt.UINT]
user32.RegisterHotKey.restype = wt.BOOL
user32.UnregisterHotKey.argtypes = [wt.HWND, ctypes.c_int]
user32.PeekMessageW.argtypes = [ctypes.POINTER(wt.MSG), wt.HWND, wt.UINT, wt.UINT, wt.UINT]
user32.PeekMessageW.restype = wt.BOOL
user32.GetWindowTextW.argtypes = [wt.HWND, wt.LPWSTR, ctypes.c_int]
user32.IsWindowVisible.argtypes = [wt.HWND]
user32.IsIconic.argtypes = [wt.HWND]
user32.ShowWindow.argtypes = [wt.HWND, ctypes.c_int]
user32.SetForegroundWindow.argtypes = [wt.HWND]
user32.GetForegroundWindow.restype = wt.HWND
user32.GetWindowThreadProcessId.argtypes = [wt.HWND, ctypes.POINTER(wt.DWORD)]
user32.GetWindowThreadProcessId.restype = wt.DWORD
user32.AttachThreadInput.argtypes = [wt.DWORD, wt.DWORD, wt.BOOL]
user32.BringWindowToTop.argtypes = [wt.HWND]
kernel32.GetCurrentThreadId.restype = wt.DWORD

_EnumProc = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
user32.EnumWindows.argtypes = [_EnumProc, wt.LPARAM]


def find_jarvis_window():
    """Returns the handle of a visible top-level window whose title contains
    WINDOW_TITLE_MARKER, or None. (Only finds it if the Jarvis tab is the
    browser's active tab -- that's the only time the title shows up.)"""
    found = []

    def _visit(hwnd, _lparam):
        if user32.IsWindowVisible(hwnd):
            buf = ctypes.create_unicode_buffer(256)
            user32.GetWindowTextW(hwnd, buf, 256)
            if WINDOW_TITLE_MARKER in buf.value:
                found.append(hwnd)
                return False  # stop enumerating
        return True

    user32.EnumWindows(_EnumProc(_visit), 0)
    return found[0] if found else None


def raise_window(hwnd):
    """Brings `hwnd` to the foreground. Windows blocks plain
    SetForegroundWindow calls from background processes, so briefly attach to
    the current foreground window's input queue first -- the standard way
    around that restriction."""
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, _SW_RESTORE)
    fg = user32.GetForegroundWindow()
    fg_thread = user32.GetWindowThreadProcessId(fg, None) if fg else 0
    this_thread = kernel32.GetCurrentThreadId()
    attached = False
    if fg_thread and fg_thread != this_thread:
        attached = bool(user32.AttachThreadInput(this_thread, fg_thread, True))
    try:
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
    finally:
        if attached:
            user32.AttachThreadInput(this_thread, fg_thread, False)


def _hotkey_thread(on_hotkey, stop_event):
    # RegisterHotKey is per-thread: the registration, the message pump and
    # the unregistration must all happen on this same thread.
    if not user32.RegisterHotKey(None, _HOTKEY_ID, HOTKEY_MODIFIERS | MOD_NOREPEAT, HOTKEY_VK):
        err = ctypes.get_last_error()
        print(f"[Global hotkey {HOTKEY_LABEL} unavailable (error {err}) -- another app "
              f"probably owns it. Edit hotkey.py to pick a different one. The in-page "
              f"shortcut still works while the Jarvis window is focused.]")
        return
    print(f"Global hotkey {HOTKEY_LABEL} active -- focuses the Jarvis input from anywhere.")
    msg = wt.MSG()
    try:
        while not stop_event.is_set():
            if user32.PeekMessageW(ctypes.byref(msg), None, _WM_HOTKEY, _WM_HOTKEY, _PM_REMOVE):
                if msg.message == _WM_HOTKEY and msg.wParam == _HOTKEY_ID:
                    try:
                        on_hotkey()
                    except Exception as exc:
                        print(f"[Hotkey handler error]: {exc}")
            else:
                time.sleep(0.02)
    finally:
        user32.UnregisterHotKey(None, _HOTKEY_ID)


def start(state, stop_event):
    """Starts the hotkey listener thread. On press: ask the UI to focus its
    input, and raise the Jarvis browser window if it can be found."""

    def on_hotkey():
        hwnd = find_jarvis_window()
        if hwnd:
            raise_window(hwnd)
        else:
            print("[Hotkey pressed, but no visible window titled "
                  f"'{WINDOW_TITLE_MARKER}' -- is the Jarvis tab active in your browser?]")
        state.request_focus()

    thread = threading.Thread(
        target=_hotkey_thread, args=(on_hotkey, stop_event), daemon=True, name="jarvis-hotkey"
    )
    thread.start()
    return thread
