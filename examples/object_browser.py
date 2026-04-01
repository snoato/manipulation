"""Interactive browser for YCB and Google Scanned Objects.

Lets you pick objects from the remote libraries, downloads them on demand,
and places them on the tabletop in a MuJoCo viewer.

Run with::

    cd examples
    mjpython object_browser.py

You can pre-select objects with ``--ycb`` and ``--gso`` flags::

    mjpython object_browser.py --ycb 002_master_chef_can 003_cracker_box --gso Alarm_Clock

Or list what's available without launching the viewer::

    python object_browser.py --list-ycb
    python object_browser.py --list-gso

Set GITHUB_TOKEN in your environment to avoid API rate-limit issues when
browsing or downloading many objects.
"""

import argparse
import sys
import textwrap
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TABLE_POS  = [0.0, 0.4, 0.0]           # matches tabletop domain (rotated 90°)
_TABLE_QUAT = [0.0, 0.0, 0.0, 1.0]     # same rotation as make_symbolic_builder
_TABLE_CX   = 0.45                      # world-frame table centre x
_TABLE_CY   = 0.60                      # world-frame table centre y
_TABLE_H    = 0.27                      # surface height (m)
_OBJECT_Z   = _TABLE_H + 0.06          # place objects above table surface


def _paginate(items, page_size=20):
    """Print items in pages, returning the full list."""
    for i, item in enumerate(items):
        print(f"  {i+1:4d}. {item}")
        if (i + 1) % page_size == 0 and i + 1 < len(items):
            ans = input(f"  -- {i+1}/{len(items)} shown. Press Enter for more (q to stop): ")
            if ans.strip().lower() == "q":
                break
    return items


def _parse_selection(raw: str, items) -> list:
    """Parse a comma-separated mix of numbers and names into object names."""
    selected = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.isdigit():
            idx = int(tok) - 1
            if 0 <= idx < len(items):
                selected.append(items[idx])
            else:
                print(f"  [warn] index {tok} out of range — skipped")
        elif tok in items:
            selected.append(tok)
        else:
            # fuzzy: prefix match
            matches = [n for n in items if n.lower().startswith(tok.lower())]
            if matches:
                selected.append(matches[0])
                if len(matches) > 1:
                    print(f"  [warn] '{tok}' matched '{matches[0]}' (also: {matches[1:3]}…)")
            else:
                print(f"  [warn] '{tok}' not found — skipped")
    return selected


def _interactive_pick(downloader, source_label: str) -> list:
    """Show available objects and let the user select some."""
    print(f"\nFetching {source_label} object list …")
    try:
        available = downloader.list_available()
    except Exception as exc:
        print(f"  [error] Could not fetch list: {exc}")
        print("  Tip: set GITHUB_TOKEN to avoid rate limits.")
        return []
    if not available:
        print("  No objects found.")
        return []
    print(f"\n  {len(available)} {source_label} objects available:\n")
    _paginate(available)
    raw = input(
        "\n  Enter numbers or names (comma-separated) to add, or press Enter to skip: "
    )
    if not raw.strip():
        return []
    return _parse_selection(raw, available)


# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------

def _build_scene(ycb_names, gso_names):
    from tampanda.scenes import SceneBuilder
    from tampanda.scenes import TABLE_SYMBOLIC_TEMPLATE

    builder = SceneBuilder()
    builder.add_resource("table", TABLE_SYMBOLIC_TEMPLATE)

    # Register and add YCB objects
    for name in ycb_names:
        builder.add_resource(f"ycb_{name}", {"type": "ycb", "name": name})

    # Register and add GSO objects
    for name in gso_names:
        builder.add_resource(f"gso_{name}", {"type": "gso", "name": name})

    # Place table (same pose as make_symbolic_builder)
    builder.add_object("table", pos=_TABLE_POS, quat=_TABLE_QUAT)

    # Place objects in a grid on the table surface
    all_objects = [(f"ycb_{n}", n) for n in ycb_names] + [(f"gso_{n}", n) for n in gso_names]
    cols = max(1, min(4, len(all_objects)))
    spacing = 0.12  # m
    for i, (res_key, name) in enumerate(all_objects):
        row = i // cols
        col = i % cols
        ox = _TABLE_CX - (cols - 1) * spacing / 2 + col * spacing
        oy = _TABLE_CY - row * spacing
        builder.add_object(res_key, pos=[ox, oy, _OBJECT_Z], name=name.replace(" ", "_"))

    return builder


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Browse and place YCB / GSO objects in a MuJoCo viewer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              mjpython object_browser.py                     # fully interactive
              mjpython object_browser.py --ycb 002_master_chef_can 003_cracker_box
              mjpython object_browser.py --gso Alarm_Clock Apple
              python  object_browser.py  --list-ycb
              python  object_browser.py  --list-gso
        """),
    )
    parser.add_argument("--ycb", nargs="*", metavar="NAME",
                        help="YCB object names to load (skips interactive prompt)")
    parser.add_argument("--gso", nargs="*", metavar="NAME",
                        help="GSO object names to load (skips interactive prompt)")
    parser.add_argument("--list-ycb", action="store_true",
                        help="Print all available YCB objects and exit")
    parser.add_argument("--list-gso", action="store_true",
                        help="Print all available GSO objects and exit")
    args = parser.parse_args()

    from tampanda.scenes.assets import YCBDownloader, GSODownloader

    ycb_dl = YCBDownloader()
    gso_dl = GSODownloader()

    # --- List-only modes ---
    if args.list_ycb:
        print("Fetching YCB object list …")
        try:
            names = ycb_dl.list_available()
            print(f"\n{len(names)} YCB objects:\n")
            for n in names:
                print(f"  {n}")
        except Exception as exc:
            print(f"Error: {exc}")
        return

    if args.list_gso:
        print("Fetching GSO object list …")
        try:
            names = gso_dl.list_available()
            print(f"\n{len(names)} GSO objects:\n")
            for n in names:
                print(f"  {n}")
        except Exception as exc:
            print(f"Error: {exc}")
        return

    # --- Interactive or command-line selection ---
    ycb_names = list(args.ycb) if args.ycb is not None else []
    gso_names = list(args.gso) if args.gso is not None else []

    if args.ycb is None and args.gso is None:
        # Fully interactive
        print("=" * 60)
        print("  Object Browser — YCB + Google Scanned Objects")
        print("=" * 60)
        print("\nObjects are downloaded on demand and cached in")
        print("  ~/.cache/manipulation/assets/")
        print("Set GITHUB_TOKEN to avoid rate limits (60 req/hr unauthenticated).")

        add_ycb = input("\nBrowse YCB objects? [y/N] ").strip().lower() == "y"
        if add_ycb:
            ycb_names = _interactive_pick(ycb_dl, "YCB")

        add_gso = input("\nBrowse Google Scanned Objects? [y/N] ").strip().lower() == "y"
        if add_gso:
            gso_names = _interactive_pick(gso_dl, "GSO")

    if not ycb_names and not gso_names:
        print("\nNo objects selected. Exiting.")
        return

    print(f"\nSelected YCB : {ycb_names or '(none)'}")
    print(f"Selected GSO : {gso_names or '(none)'}")
    print("\nBuilding scene (downloads as needed) …")

    try:
        builder = _build_scene(ycb_names, gso_names)
        env = builder.build_env(rate=200.0)
    except Exception as exc:
        print(f"\n[error] Scene construction failed: {exc}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    print("\nLaunching viewer — close the window to exit.")

    import mujoco
    import mujoco.viewer

    with mujoco.viewer.launch_passive(
        model=env.model, data=env.data,
        show_left_ui=False, show_right_ui=False,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(env.model, viewer.cam)
        viewer.sync()
        while viewer.is_running():
            mujoco.mj_step(env.model, env.data)
            viewer.sync()


if __name__ == "__main__":
    main()
