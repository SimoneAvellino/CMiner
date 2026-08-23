"""
Disk spill for pattern occurrence mappings.

Goal: bound the RAM held by occurrence mappings (Mapping objects) during
mining, so long runs on large databases do not OOM. Occurrence lists that
are not currently in use are serialized to per-pattern temp files (pickle)
and transparently reloaded on access.

Why this stays EXACT (lossless):

- The search algorithm is untouched; only *where* occurrence lists live
  changes. The same Mapping objects are written out and read back.
- A PatternMappings becomes evictable only after ``seal()`` (i.e. once it
  is fully materialized; sealed instances are immutable) and only while it
  is not ``pin()``-ed.
- Identity-keyed lookups on Mapping objects (``Extension.location`` dicts
  and the ``target_map not in ...`` membership test in
  ``update_edge_mappings``) are created and consumed within a single
  pattern-processing step. The mining loops ``pin()`` the involved
  PatternMappings for the whole step, so a spilled-then-reloaded copy
  (which would have different object identities) can never be mixed with
  stale identity references.
- Per-graph occurrence counts are kept in RAM at all times, so
  ``support()`` / ``frequency()`` never force a reload.

The feature is driven by a process-wide singleton configured through
``configure_spill()`` (called by ``CMinerAPI``). When no manager is
configured (e.g. the legacy ``CMiner`` class / CLI entry point),
``get_spill_manager()`` returns None and ``PatternMappings`` behaves
exactly as before (pure in-RAM, zero overhead).

Environment overrides (read by ``configure_spill``):
    CMINER_SPILL            1/0 to force-enable/disable (default: enabled)
    CMINER_SPILL_MAX_LINKS  loaded-link budget before eviction (int)
    CMINER_SPILL_DIR        directory for spill files (default: temp dir)
    CMINER_SPILL_DEBUG      1 to log spill/load/evict events to stderr
"""

import atexit
import os
import pickle
import shutil
import sys
import tempfile
import threading
import weakref
from collections import OrderedDict

DEFAULT_MAX_LOADED_LINKS = 4_000_000


def _debug_enabled() -> bool:
    return os.environ.get("CMINER_SPILL_DEBUG", "0") == "1"


def _log(msg: str):
    if _debug_enabled():
        print(f"[cminer-spill] {msg}", file=sys.stderr)


class SpillManager:
    """
    Owns the spill directory and the LRU of sealed PatternMappings.

    The manager tracks a budget of "loaded links" (one link = one Mapping
    object in an occurrence chain; estimated as occurrences x pattern
    depth). When the budget is exceeded, least-recently-sealed/loaded,
    non-pinned PatternMappings are spilled to disk until the budget is met
    (or everything spillable is already spilled).
    """

    def __init__(self, max_loaded_links: int = DEFAULT_MAX_LOADED_LINKS, directory: str | None = None):
        self.max_loaded_links = max(1, int(max_loaded_links))
        self._own_dir = directory is None
        self.dir = directory or tempfile.mkdtemp(prefix="cminer_spill_")
        os.makedirs(self.dir, exist_ok=True)
        self._lock = threading.RLock()
        # LRU: pm_uid -> (weakref(pm), weight). Most-recent at the end.
        self._lru: OrderedDict[int, tuple[weakref.ref, int]] = OrderedDict()
        self._loaded_links = 0
        self._uids = 0
        self.stats = {"spills": 0, "loads": 0, "bytes_written": 0, "bytes_read": 0}

    # ---- naming ----

    def next_uid(self) -> int:
        with self._lock:
            self._uids += 1
            return self._uids

    def file_for(self, pm_uid: int) -> str:
        return os.path.join(self.dir, f"pm_{pm_uid:08d}.pkl")

    # ---- lifecycle ----

    def register(self, pm, weight: int):
        """
        Seal-time registration of a fully in-RAM PatternMappings.
        Triggers eviction if the loaded-link budget is exceeded.
        """
        with self._lock:
            pm._weight = weight
            self._lru[pm._uid] = (weakref.ref(pm), weight)
            self._lru.move_to_end(pm._uid)
            self._loaded_links += weight
            weakref.finalize(pm, SpillManager._on_dead, weakref.ref(self), pm._uid)
            _log(
                f"register pm#{pm._uid} weight={weight} "
                f"loaded={self._loaded_links}/{self.max_loaded_links}"
            )
            self._evict_locked(exempt_uid=pm._uid)

    def note_loaded(self, pm):
        """A spilled PatternMappings was reloaded into RAM."""
        with self._lock:
            self._lru[pm._uid] = (weakref.ref(pm), pm._weight)
            self._lru.move_to_end(pm._uid)
            self._loaded_links += pm._weight
            _log(
                f"loaded pm#{pm._uid} weight={pm._weight} "
                f"loaded={self._loaded_links}/{self.max_loaded_links}"
            )
            self._evict_locked(exempt_uid=pm._uid)

    def note_spilled(self, pm):
        """A PatternMappings dropped its in-RAM lists (data is on disk)."""
        with self._lock:
            entry = self._lru.pop(pm._uid, None)
            if entry is not None:
                self._loaded_links -= entry[1]
            self.stats["spills"] += 1

    @staticmethod
    def _on_dead(manager_ref, pm_uid: int):
        """weakref finalizer: purge LRU entry and spill file of a dead pm."""
        manager = manager_ref()
        if manager is None:
            return
        with manager._lock:
            entry = manager._lru.pop(pm_uid, None)
            if entry is not None:
                manager._loaded_links -= entry[1]
            try:
                os.remove(manager.file_for(pm_uid))
            except OSError:
                pass

    # ---- eviction ----

    def _evict_locked(self, exempt_uid: int | None = None):
        """
        Spill LRU (least-recently registered/loaded), non-pinned
        PatternMappings until the loaded-link budget is met.
        Caller must hold ``self._lock`` (RLock: pm spill callbacks re-enter).
        """
        if self._loaded_links <= self.max_loaded_links:
            return
        skipped = 0
        while self._loaded_links > self.max_loaded_links and self._lru:
            uid, (ref, _weight) = next(iter(self._lru.items()))  # LRU end
            pm = ref() if ref is not None else None
            if pm is None:
                entry = self._lru.pop(uid, None)
                if entry is not None:
                    self._loaded_links -= entry[1]
                continue
            if uid == exempt_uid or pm._pinned > 0:
                # Not evictable right now: mark MRU and try the next one.
                self._lru.move_to_end(uid)
                skipped += 1
                if skipped >= len(self._lru):
                    break  # everything is pinned/exempt: nothing to evict
                continue
            _log(f"evict pm#{uid} weight={pm._weight}")
            if pm._spill_now():  # re-enters via note_spilled (RLock)
                skipped = 0
            else:
                # Lost a pin race between the check above and the spill:
                # treat as not evictable right now and move on.
                self._lru.move_to_end(uid)
                skipped += 1
                if skipped >= len(self._lru):
                    break

    # ---- cleanup ----

    def close(self):
        """Remove the spill directory (best effort)."""
        with self._lock:
            try:
                if self._own_dir:
                    shutil.rmtree(self.dir, ignore_errors=True)
                else:
                    for name in os.listdir(self.dir):
                        if name.startswith("pm_") and name.endswith(".pkl"):
                            try:
                                os.remove(os.path.join(self.dir, name))
                            except OSError:
                                pass
            except OSError:
                pass
            self._lru.clear()
            self._loaded_links = 0
            _log(
                f"closed (spills={self.stats['spills']} "
                f"loads={self.stats['loads']} "
                f"MB_written={self.stats['bytes_written'] / 1e6:.1f} "
                f"MB_read={self.stats['bytes_read'] / 1e6:.1f})"
            )


# ---- process-wide singleton ----

_manager: SpillManager | None = None
_config: tuple[bool, int, str | None] | None = None
_manager_lock = threading.Lock()


def configure_spill(
    enabled: bool = True,
    max_loaded: int = DEFAULT_MAX_LOADED_LINKS,
    directory: str | None = None,
):
    """
    Configure the process-wide spill manager. Called by CMinerAPI.__init__.
    Environment variables override the arguments (see module docstring).
    Reconfiguring with different parameters closes the previous manager.
    """
    global _config
    env_on = os.environ.get("CMINER_SPILL")
    if env_on is not None:
        enabled = env_on.strip().lower() not in ("0", "false", "no", "off")
    env_max = os.environ.get("CMINER_SPILL_MAX_LINKS")
    if env_max is not None:
        try:
            max_loaded = int(env_max)
        except ValueError:
            pass
    env_dir = os.environ.get("CMINER_SPILL_DIR")
    if env_dir:
        directory = env_dir

    new_config = (bool(enabled), int(max_loaded), directory)
    with _manager_lock:
        global _manager
        if _config == new_config:
            return
        _config = new_config
        if _manager is not None:
            _manager.close()
            _manager = None
        if enabled:
            _manager = SpillManager(max_loaded_links=max_loaded, directory=directory)
            _log(
                f"configured: max_loaded_links={max_loaded} dir={_manager.dir}"
            )


def get_spill_manager() -> SpillManager | None:
    """
    Return the active SpillManager, or None when spilling is disabled.
    Lazily recreates the manager if it was closed while the config is enabled.
    """
    global _manager
    with _manager_lock:
        if _config is None or not _config[0]:
            return None
        if _manager is None:
            _manager = SpillManager(max_loaded_links=_config[1], directory=_config[2])
        return _manager


def close_spill_manager():
    """Close the manager and wipe spill files. The config is kept, so the
    next get_spill_manager()/configure_spill() recreates a fresh manager."""
    global _manager
    with _manager_lock:
        if _manager is not None:
            _manager.close()
            _manager = None


@atexit.register
def _cleanup_at_exit():
    close_spill_manager()
