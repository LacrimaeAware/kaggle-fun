from __future__ import annotations

import deck_policy as DP
import main


def obs_with(select, me=None, opp=None, yi=0):
    return {
        'current': {
            'yourIndex': yi,
            'players': [me or {}, opp or {}],
        },
        'select': select,
    }


def test_powerful_hand_dynamic():
    me = {'active': [{'id': 743, 'hp': 140}], 'bench': [], 'handCount': 7, 'prize': [1, 2]}
    opp = {'active': [{'id': 999, 'hp': 130}], 'bench': []}
    o = {'type': 13, 'attackId': 123}
    obs = obs_with({'maxCount': 1, 'minCount': 1, 'option': [o]}, me, opp)
    assert DP.attack_damage(o, obs) == 140
    assert DP.attack_value(o, obs) >= 8000


def test_only_final_ko_forced():
    # Dynamic Powerful Hand KOs, but with two prizes remaining it must fall through to search.
    me = {'active': [{'id': 743, 'hp': 140}], 'bench': [], 'handCount': 7, 'prize': [1, 2]}
    opp = {'active': [{'id': 999, 'hp': 130}], 'bench': []}
    o = {'type': 13, 'attackId': 123}
    obs = obs_with({'maxCount': 1, 'minCount': 1, 'option': [o]}, me, opp)
    assert main._forced_move(obs) is None
    me['prize'] = [1]
    assert main._forced_move(obs) == [0]


def test_tutor_prefers_alakazam_completion():
    me = {
        'active': [{'id': 742, 'hp': 80}],
        'bench': [],
        'hand': [],
        'deck': [{'id': 741}, {'id': 743}, {'id': 5}],
        'prize': [1, 2, 3],
    }
    opp = {'active': [{'id': 999, 'hp': 200}], 'bench': []}
    sel = {
        'maxCount': 1,
        'minCount': 1,
        'option': [
            {'type': 3, 'area': 1, 'index': 0, 'playerIndex': 0},
            {'type': 3, 'area': 1, 'index': 1, 'playerIndex': 0},
            {'type': 3, 'area': 1, 'index': 2, 'playerIndex': 0},
        ],
    }
    assert DP.choose_subdecision(obs_with(sel, me, opp)) == [1]


def test_optional_prompt_can_decline():
    me = {'active': [{'id': 743, 'hp': 140}], 'bench': [], 'hand': [], 'deck': [{'id': 999}]}
    opp = {'active': [{'id': 998, 'hp': 200}], 'bench': []}
    sel = {'maxCount': 1, 'minCount': 0, 'option': [{'type': 3, 'area': 1, 'index': 0, 'playerIndex': 0}]}
    # Unknown card gets zero score; optional prompt should be declined.
    assert DP.choose_subdecision(obs_with(sel, me, opp)) == []


def test_root_search_not_overridden():
    me = {'active': [{'id': 743, 'hp': 140}], 'bench': [], 'hand': [{'id': 1231}]}
    opp = {'active': [{'id': 998, 'hp': 200}], 'bench': []}
    sel = {'maxCount': 1, 'minCount': 1, 'option': [{'type': 7, 'index': 0}, {'type': 14}]}
    assert DP.choose_subdecision(obs_with(sel, me, opp)) is None
    priors = DP.root_option_priors(obs_with(sel, me, opp))
    assert priors is not None and len(priors) == 2


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in tests:
        fn()
    print(f'{len(tests)} tests passed')
