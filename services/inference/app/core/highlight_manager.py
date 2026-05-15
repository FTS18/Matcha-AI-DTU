import logging

logger = logging.getLogger(__name__)


def select_highlights(scored_events, duration, top_n=5, clip_secs=30.0):
    """
    Top-N non-overlapping highlights, spread across the full video.
    Two rules:
    1. No time-window overlap between clips.
    2. Clip centres must be at least 15% of duration apart (prevents same-scene
    from different YOLO frames appearing twice).
    """
    if not scored_events:
        return []
    sorted_evs = sorted(scored_events, key=lambda x: x["finalScore"], reverse=True)
    min_spread = max(30.0, duration * 0.15)  # at least 15% of the video length
    used, highlights = [], []

    for ev in sorted_evs:
        if len(highlights) >= top_n:
            break
        ts = ev["timestamp"]
        start = max(0.0, ts - clip_secs * 0.35)
        end = min(duration if duration > 0 else ts + 60.0, ts + clip_secs * 0.65)

        # Rule 1 \u2013 no overlap
        if any(not (end <= w[0] or start >= w[1]) for w in used):
            continue

        # Rule 2 \u2013 clips must be spread out
        if any(
            abs(ts - (h["startTime"] + (h["endTime"] - h["startTime"]) / 2))
            < min_spread
            for h in highlights
        ):
            continue

        used.append((start, end))
        highlights.append(
            {
                "startTime": round(start, 1),
                "endTime": round(end, 1),
                "score": ev["finalScore"],
                "eventType": ev["type"],
                "commentary": ev.get("commentary", ""),
            }
        )

    return highlights


def group_related_events(scored_events: list, min_gap_secs: float = 15.0) -> list:
    """
    Group related events (e.g., build-up + goal) for better narrative flow.
    Returns events with group_id for clustering.
    """
    if not scored_events:
        return []

    grouped = []
    current_group = 0
    last_group_time = scored_events[0]["timestamp"]

    for event in scored_events:
        time_since_group = event["timestamp"] - last_group_time

        # Start new group if gap is large
        if time_since_group > min_gap_secs:
            current_group += 1
            last_group_time = event["timestamp"]

        event_copy = event.copy()
        event_copy["group_id"] = current_group
        event_copy["time_in_group"] = round(time_since_group, 2)
        grouped.append(event_copy)

    return grouped


def select_highlights_with_narrative(
    scored_events: list,
    duration: float,
    top_n: int = 5,
    clip_secs: float = 30.0,
    use_groups: bool = True,
) -> list:
    """
    Enhanced highlight selection that considers narrative flow and event grouping.
    Groups build-up sequences with their payoff (e.g., goal sequences).
    """
    if not scored_events:
        return []

    # Group related events if enabled
    if use_groups:
        grouped_events = group_related_events(scored_events, min_gap_secs=15.0)
    else:
        grouped_events = scored_events

    # Sort by final score
    sorted_evs = sorted(grouped_events, key=lambda x: x["finalScore"], reverse=True)

    min_spread = max(30.0, duration * 0.15)
    used = []
    highlights = []

    for ev in sorted_evs:
        if len(highlights) >= top_n:
            break

        ts = ev["timestamp"]

        # Extend clip backwards if there's a group with lead-up events
        start_buffer = (
            clip_secs * 0.50
            if "group_id" in ev and ev.get("time_in_group", 0) > 10
            else clip_secs * 0.35
        )

        start = max(0.0, ts - start_buffer)
        end = min(duration if duration > 0 else ts + 60.0, ts + clip_secs * 0.65)

        # Rule 1: No overlap
        if any(not (end <= w[0] or start >= w[1]) for w in used):
            continue

        # Rule 2: Clips spread out
        if any(
            abs(ts - (h["startTime"] + (h["endTime"] - h["startTime"]) / 2))
            < min_spread
            for h in highlights
        ):
            continue

        used.append((start, end))
        highlights.append(
            {
                "startTime": round(start, 1),
                "endTime": round(end, 1),
                "score": ev["finalScore"],
                "eventType": ev["type"],
                "commentary": ev.get("commentary", ""),
                "group_id": ev.get("group_id", -1),
                "narrative_context": True if use_groups else False,
            }
        )

    return sorted(highlights, key=lambda x: x["startTime"])
