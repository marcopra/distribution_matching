from unittest.mock import Mock

from gym_env import ActionRepeatWrapper


def make_tracker(room_ids):
    tracker = object.__new__(ActionRepeatWrapper)
    tracker._is_montezuma = True
    tracker._get_montezuma_room_id = Mock(side_effect=room_ids)
    tracker._montezuma_initial_room = None
    tracker._montezuma_escaped_first_room = False
    tracker._montezuma_rooms_visited = set()
    tracker._montezuma_room_route = []
    tracker._montezuma_episode_frame = 0
    return tracker


def test_montezuma_room_metrics_track_distinct_rooms_and_returns():
    tracker = make_tracker([1, 1, 7, 7, 1, 3])

    assert tracker._reset_montezuma_tracking() == 1
    for _ in range(5):
        tracker._update_montezuma_tracking()

    info = tracker._augment_info({}, 3)
    assert info['montezuma_escaped_first_room'] is True
    assert info['montezuma_unique_rooms_visited'] == 3
    assert info['montezuma_room_transition_count'] == 3
    assert info['montezuma_room_route'] == (
        (0, 1),
        (2, 7),
        (4, 1),
        (5, 3),
    )
    assert 'montezuma_max_room_id' not in info


def test_montezuma_room_metrics_reset_between_episodes():
    tracker = make_tracker([1, 7, 7])
    tracker._reset_montezuma_tracking()
    tracker._update_montezuma_tracking()

    assert tracker._reset_montezuma_tracking() == 7
    info = tracker._augment_info({}, 7)
    assert info['montezuma_escaped_first_room'] is False
    assert info['montezuma_unique_rooms_visited'] == 1
    assert info['montezuma_room_transition_count'] == 0
    assert info['montezuma_room_route'] == ((0, 7),)
