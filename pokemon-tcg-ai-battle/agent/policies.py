"""Policies for the mock environment.

Two policies share one interface: given a GameState (and the env), return a legal Action.

  RandomLegalPolicy   uniform over legal actions. The correctness baseline.
  HeuristicPolicy     one-ply greedy: try each legal action, evaluate the resulting
                      state, take the best. Always legal, never loops (each turn ends in
                      an attack or a pass), cheap (no recursion), so it cannot time out.

The heuristic is intentionally simple and explainable. Every term has a reason. When we
have the real engine, this same policy runs unchanged against it: only the evaluation
weights and the legal-action source change. The point of the first attempt is a robust
agent that always plays a legal move and beats random by a wide margin, not a clever one.

The evaluation is from the moving player's point of view. The dominant term is the prize
race (you win by taking prizes); board hit points, energy on the attacker, bench width,
and hand size are tie-breakers that encourage setup before attacking.
"""
from __future__ import annotations

import random

from ptcg_mock import GameState, MockPTCG, Action

WIN = 1000.0


def evaluate_state(state: GameState, player: int) -> float:
    if state.winner is not None:
        return WIN if state.winner == player else -WIN
    me, opp = state.players[player], state.players[1 - player]

    def board_hp(p) -> int:
        total = me_active_hp = 0
        if p.active:
            total += p.active.hp_left
        for b in p.bench:
            total += b.hp_left
        return total

    # prize race dominates: I win when my prizes_remaining hits 0, so lower is better.
    prize = (opp.prizes_remaining - me.prizes_remaining) * 50.0
    hp = (board_hp(me) - board_hp(opp)) * 0.3
    my_e = me.active.energy if me.active else 0
    opp_e = opp.active.energy if opp.active else 0
    energy = (my_e - opp_e) * 3.0
    bench = (len(me.bench) - len(opp.bench)) * 2.0
    hand = (len(me.hand) - len(opp.hand)) * 0.5
    # progress toward a knockout on the opponent's active
    chip = (opp.active.damage if opp.active else 0) * 0.3
    return prize + hp + energy + bench + hand + chip


# tie-break preference so the agent sets up, then attacks, and never loops within a turn
_PREF = {"attack": 4, "attach": 3, "play_basic": 2, "retreat": 1, "pass": 0}


class RandomLegalPolicy:
    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def act(self, env: MockPTCG, state: GameState) -> Action:
        return self.rng.choice(env.legal_actions(state))


class HeuristicPolicy:
    """One-ply greedy over legal actions. Deterministic given the state."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def act(self, env: MockPTCG, state: GameState) -> Action:
        player = state.to_move
        legal = env.legal_actions(state)
        base = evaluate_state(state, player)

        # Attacking ends the turn, so do every beneficial NON-terminal action first
        # (attach energy, develop the bench, retreat into a better attacker). Only end
        # the turn once setup no longer improves the position. Without this ordering the
        # agent would attack on its first sub-action and never set up.
        setup = [a for a in legal if a[0] in ("attach", "play_basic", "retreat")]
        best_a, best_delta = None, 1e-9
        for a in setup:
            delta = evaluate_state(env.step(state, a), player) - base
            if delta > best_delta:
                best_delta, best_a = delta, a
        if best_a is not None:
            return best_a

        # Setup exhausted: take the best attack if one is affordable, else end the turn.
        attacks = [a for a in legal if a[0] == "attack"]
        if attacks:
            return max(attacks, key=lambda a: evaluate_state(env.step(state, a), player))
        if any(a[0] == "pass" for a in legal):
            return ("pass",)
        return legal[0]      # forced move (e.g. must promote an active)
