from __future__ import annotations

from .tracker import Phase, TrajectoryTracker


def sync_stereo_awaiting(main: TrajectoryTracker, secondary: TrajectoryTracker) -> None:
    """Return both trackers to idle when each has a valid completed throw."""
    if (
        main.phase == Phase.AWAITING_PARTNER
        and secondary.phase == Phase.AWAITING_PARTNER
        and main._awaiting_valid_completion
        and secondary._awaiting_valid_completion
    ):
        main.exit_awaiting_partner()
        secondary.exit_awaiting_partner()


def reconcile_stereo_trackers(
    main: TrajectoryTracker,
    secondary: TrajectoryTracker,
    *,
    throw_label: int,
    wrist_pos: tuple[int, int] | None,
) -> tuple[bool, bool]:
    """
    After per-camera updates, align cameras that failed a throw (discarded
    trajectory or scan timeout) with the partner's active phase.

    Returns ``(main_reconciled, secondary_reconciled)`` for pending-track cleanup.
    """
    main_reconciled = False
    secondary_reconciled = False

    if main.pop_stereo_reconcile():
        main.adopt_partner_phase(
            secondary,
            is_secondary=False,
            throw_label=throw_label,
            wrist_pos=wrist_pos,
        )
        main_reconciled = True

    if secondary.pop_stereo_reconcile():
        secondary.adopt_partner_phase(
            main,
            is_secondary=True,
            throw_label=throw_label,
            wrist_pos=wrist_pos,
        )
        secondary_reconciled = True

    sync_stereo_awaiting(main, secondary)
    return main_reconciled, secondary_reconciled
