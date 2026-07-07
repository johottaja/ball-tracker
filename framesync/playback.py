from __future__ import annotations

from .engine import FrameSyncEngine


def prepare_framesync_for_frame(
    engine: FrameSyncEngine,
    frame_index: int,
    last_frame_index: int | None,
    cache: object | None,
) -> int:
    """
    Reset the engine on non-sequential playback steps and restore the latest
    cached offset at or before ``frame_index``.
    """
    if last_frame_index is not None and frame_index != last_frame_index + 1:
        engine.reset()
        if cache is not None:
            event = cache.latest_sync_at_or_before(frame_index)
            if event is not None:
                engine.restore_persisted_sync(
                    offset=event.offset,
                    sync_id=event.sync_id,
                )
    elif last_frame_index is None and cache is not None:
        event = cache.latest_sync_at_or_before(frame_index)
        if event is not None:
            engine.restore_persisted_sync(
                offset=event.offset,
                sync_id=event.sync_id,
            )
    return frame_index


def record_framesync_completion(
    engine: FrameSyncEngine,
    frame_index: int,
    sync_id_before: int,
    cache: object | None,
) -> None:
    if cache is None:
        return
    offset = engine.latest_offset
    if offset is None or engine.sync_id <= sync_id_before:
        return
    cache.put_sync_event(frame_index, engine.sync_id, offset)
