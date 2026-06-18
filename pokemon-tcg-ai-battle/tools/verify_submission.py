"""Verify a built submission the way Kaggle actually loads it.

Kaggle's kaggle_environments.agent.get_last_callable does `exec(code_object, env)` with a FRESH
namespace that has NO __file__, then takes the LAST callable. A normal `import main` test does NOT
reproduce this (imported modules have __file__), which is how a __file__ reference in main.py
shipped and failed validation with ERROR on 2026-06-17. This test execs main.py exactly that way,
takes the last callable, and plays games. Exit non-zero on any load error or in-game exception.

    python tools/verify_submission.py submissions/sub_search [games]
"""
import contextlib
import io
import logging
import os
import sys

logging.disable(logging.CRITICAL)


def main() -> None:
    sub = os.path.abspath(sys.argv[1])
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    sys.path.insert(0, sub)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import kaggle_environments.envs.cabt.cabt as cabt
        from kaggle_environments import make

    src = open(os.path.join(sub, "main.py"), encoding="utf-8").read()
    env: dict = {}                                  # no __file__, exactly like Kaggle
    try:
        exec(compile(src, "main.py", "exec"), env)
    except Exception as e:
        print(f"verify {os.path.basename(sub)}: LOAD FAILED (this is what Kaggle would see): {e!r}")
        sys.exit(1)
    callables = [v for v in env.values() if callable(v)]
    if not callables:
        print(f"verify {os.path.basename(sub)}: no callable defined in main.py")
        sys.exit(1)
    agent = callables[-1]

    def winner(e):
        last = e.steps[-1]
        r0, r1 = last[0].get("reward"), last[1].get("reward")
        return None if (r0 is None or r1 is None or r0 == r1) else (0 if r0 > r1 else 1)

    wins = errors = 0
    for _ in range(games):
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                e = make("cabt")
                e.run([agent, cabt.first_agent])
            if winner(e) == 0:
                wins += 1
        except Exception:
            errors += 1
    print(f"verify {os.path.basename(sub)}: last-callable '{getattr(agent, '__name__', '?')}', "
          f"{games} games vs first, wins {wins}, errors {errors}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
