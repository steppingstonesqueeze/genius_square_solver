import streamlit as st
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import statistics

# -----------------------------
# Piece definitions & geometry
# -----------------------------

Coord = Tuple[int, int]

def normalize(shape: Set[Coord]) -> Set[Coord]:
    min_r = min(r for r, c in shape)
    min_c = min(c for r, c in shape)
    return {(r - min_r, c - min_c) for (r, c) in shape}

def rot90(shape: Set[Coord]) -> Set[Coord]:
    # (r, c) -> (c, -r)
    return normalize({(c, -r) for (r, c) in shape})

def flip(shape: Set[Coord]) -> Set[Coord]:
    # Reflect horizontally: (r, c) -> (r, -c)
    return normalize({(r, -c) for (r, c) in shape})

def unique_orientations(base: Set[Coord]) -> List[Set[Coord]]:
    # Generate all rotations and flips; deduplicate by normalized frozenset
    seen = set()
    out = []
    variants = []
    s = normalize(base)
    variants.append(s)
    r1 = rot90(s)
    r2 = rot90(r1)
    r3 = rot90(r2)
    variants.extend([r1, r2, r3])
    f = flip(s)
    variants.append(f)
    r1 = rot90(f)
    r2 = rot90(r1)
    r3 = rot90(r2)
    variants.extend([r1, r2, r3])
    for v in variants:
        key = frozenset(v)
        if key not in seen:
            seen.add(key)
            out.append(v)
    # Sort by a canonical tuple for determinism
    out.sort(key=lambda sh: sorted(list(sh)))
    return out

@dataclass(frozen=True)
class Piece:
    key: str           # short key (e.g. 'I4', 'O4', ...)
    label: str         # human-readable
    cells: Set[Coord]  # base shape (normalized from 0,0)
    color: str         # emoji color block used in render
    weight: int        # size (# cells), used for ordering

# Define the 9 Genius Square pieces (one of each).
# Based on community + replacement part listings:
#   Tetrominoes: I, O, T, Z, L  (5 pieces * 4 = 20)
#   Triominoes:  straight-3, L-3 (2 pieces * 3 = 6)
#   Domino: 2 (1 piece * 2 = 2)
#   Monomino: 1 (1 piece * 1 = 1)
# Total cells: 20 + 6 + 2 + 1 = 29 cells, matching 36 - 7 blockers.
#
# We allow rotation and mirror for all pieces.

PIECES: List[Piece] = [
    Piece("I4", "I (4-long line)", {(0,0),(0,1),(0,2),(0,3)}, "⬜", 4),
    Piece("O4", "O (2×2 square)", {(0,0),(0,1),(1,0),(1,1)}, "🟩", 4),
    Piece("T4", "T tetromino", {(0,0),(0,1),(0,2),(1,1)}, "🟨", 4),
    Piece("Z4", "Z tetromino", {(0,0),(0,1),(1,1),(1,2)}, "🟥", 4),
    Piece("L4", "L tetromino", {(0,0),(1,0),(2,0),(2,1)}, "🟧", 4),
    Piece("I3", "3-line triomino", {(0,0),(0,1),(0,2)}, "🟪", 3),
    Piece("L3", "L triomino", {(0,0),(1,0),(1,1)}, "🟦", 3),
    Piece("D2", "Domino (2)", {(0,0),(0,1)}, "🟫", 2),
    Piece("S1", "Single (1)", {(0,0)}, "⬛", 1),
]

PIECE_ORIENTS: Dict[str, List[Set[Coord]]] = {p.key: unique_orientations(p.cells) for p in PIECES}
PIECE_MAP = {p.key: p for p in PIECES}

BOARD_N = 6
ALL_CELLS: Set[Coord] = {(r, c) for r in range(BOARD_N) for c in range(BOARD_N)}

# -----------------------------
# Solver (backtracking, all solutions)
# -----------------------------

def placements_for_piece(piece_key: str, empty: Set[Coord]) -> List[Set[Coord]]:
    """All placements of piece on the board that fit entirely within 'empty' cells."""
    res = []
    for orient in PIECE_ORIENTS[piece_key]:
        max_r = max(r for r, _ in orient)
        max_c = max(c for _, c in orient)
        for r0 in range(BOARD_N - max_r):
            for c0 in range(BOARD_N - max_c):
                placed = {(r0 + r, c0 + c) for (r, c) in orient}
                if placed.issubset(empty):
                    res.append(placed)
    # Sort by top-left-most cell to get determinism
    res.sort(key=lambda ps: sorted(ps))
    return res

def choose_next_piece(pieces_left: List[str], empty: Set[Coord]) -> Tuple[str, List[Set[Coord]]]:
    # Heuristic: choose piece with fewest legal placements (MRV) to prune faster
    best_piece = None
    best_opts = None
    best_count = 10**9
    for pk in pieces_left:
        opts = placements_for_piece(pk, empty)
        if len(opts) < best_count:
            best_count = len(opts)
            best_piece = pk
            best_opts = opts
            if best_count == 0:
                break
    return best_piece, best_opts or []

def solve_all(blockers: Set[Coord], max_solutions: int = None) -> List[Dict[str, Set[Coord]]]:
    """Return all solutions as list of dict piece_key -> set of cells. If max_solutions set, cut off early."""
    empty = ALL_CELLS - set(blockers)
    # Quick check: cell count must be 29
    if len(empty) != 29:
        return []
    # Order pieces primarily by size desc (tie-break by key) to speed
    pieces_left = sorted([p.key for p in PIECES], key=lambda k: (-PIECE_MAP[k].weight, k))

    solutions: List[Dict[str, Set[Coord]]] = []

    def backtrack(empty_cells: Set[Coord], remaining: List[str], placed: Dict[str, Set[Coord]]):
        nonlocal solutions, max_solutions
        if not remaining:
            solutions.append({k: set(v) for k, v in placed.items()})
            return
        pk, options = choose_next_piece(remaining, empty_cells)
        if not options:
            return
        new_remaining = [x for x in remaining if x != pk]
        for opt in options:
            # place
            backtrack(empty_cells - opt, new_remaining, {**placed, pk: opt})
            if max_solutions is not None and len(solutions) >= max_solutions:
                return

    backtrack(empty, pieces_left, {})
    return solutions

# -----------------------------
# UI helpers
# -----------------------------

def coord_label(r: int, c: int) -> str:
    return f"{chr(ord('A')+c)}{r+1}"

def render_grid_interactive(blockers: Set[Coord], placed: Dict[str, Set[Coord]], mode: str, selected_piece: str = None, orient_idx: int = 0):
    """Render a 6x6 grid with clickable cells. 
       mode: 'blockers', 'place', 'remove'
    """
    grid_owner = {}  # (r,c) -> piece_key or 'BLOCK'
    for (r, c) in blockers:
        grid_owner[(r, c)] = 'BLOCK'
    for pk, cells in placed.items():
        for rc in cells:
            grid_owner[rc] = pk

    # Compute the shape preview for selected piece at hovered cell? (streamlit has no hover info)
    # We'll just show the current shape below the grid.

    cell_size_css = """
    <style>
    .cell-btn { width: 42px !important; height: 42px !important; padding: 0 !important; }
    .small { font-size: 14px; }
    </style>
    """
    st.markdown(cell_size_css, unsafe_allow_html=True)

    for r in range(BOARD_N):
        cols = st.columns(BOARD_N, gap="small")
        for c in range(BOARD_N):
            owner = grid_owner.get((r, c))
            label = " "
            if owner == 'BLOCK':
                label = "⛔"
            elif owner is not None:
                label = PIECE_MAP[owner].color
            key = f"cell_{r}_{c}"
            clicked = cols[c].button(label, key=key, help=coord_label(r,c), use_container_width=True)
            if clicked:
                if mode == 'blockers':
                    # toggle blocker
                    if (r, c) in blockers:
                        blockers.remove((r, c))
                    else:
                        if len(blockers) < 7:
                            blockers.add((r, c))
                elif mode == 'place' and selected_piece:
                    try_place_piece(selected_piece, (r, c), orient_idx)
                elif mode == 'remove':
                    # remove piece if any
                    if (r, c) in grid_owner and grid_owner[(r, c)] not in (None, 'BLOCK'):
                        pk = grid_owner[(r, c)]
                        del st.session_state.placed[pk]
                st.rerun()

def try_place_piece(piece_key: str, anchor: Coord, orient_idx: int):
    orient = PIECE_ORIENTS[piece_key][orient_idx % len(PIECE_ORIENTS[piece_key])]
    ar, ac = anchor
    placed = {(ar + r, ac + c) for (r, c) in orient}
    # Check bounds
    if any(r < 0 or r >= BOARD_N or c < 0 or c >= BOARD_N for (r, c) in placed):
        st.toast("❌ Doesn't fit on board.")
        return
    # Check overlap with blockers or other pieces
    occupied = set(st.session_state.blockers)
    for pk, cells in st.session_state.placed.items():
        occupied |= set(cells)
    if placed & occupied:
        st.toast("❌ Overlaps something. Try a different spot or rotate/flip.")
        return
    # Place (replace prior placement if re-placing same piece)
    st.session_state.placed[piece_key] = placed

def render_piece_palette():
    st.write("### Pieces")
    cols = st.columns(3, gap="small")
    for i, p in enumerate(PIECES):
        with cols[i % 3]:
            placed = p.key in st.session_state.placed
            st.write(f"{p.color} **{p.key}** – {p.label} ({p.weight})")
            if placed:
                if st.button(f"🗑 Remove {p.key}", key=f"rm_{p.key}"):
                    del st.session_state.placed[p.key]
                    st.rerun()
            else:
                if st.button(f"🎯 Select {p.key}", key=f"sel_{p.key}"):
                    st.session_state.selected_piece = p.key
                    st.session_state.orient_idx = 0
                    st.session_state.mode = 'place'
                    st.rerun()

def render_selected_piece_controls():
    pk = st.session_state.selected_piece
    if not pk:
        return
    orients = PIECE_ORIENTS[pk]
    st.write(f"#### Selected: {PIECE_MAP[pk].color} **{pk}** – {PIECE_MAP[pk].label}")
    cols = st.columns(4)
    if cols[0].button("⟲ Rotate", key="rot_left"):
        # rotate = move to next orientation (we don't need a separate left/right; any change cycles)
        st.session_state.orient_idx = (st.session_state.orient_idx + 1) % len(orients)
        st.rerun()
    if cols[1].button("⟲⟲ Rotate x2", key="rot_2"):
        st.session_state.orient_idx = (st.session_state.orient_idx + 2) % len(orients)
        st.rerun()
    if cols[2].button("⇋ Flip", key="flip"):
        # flipping is already included in orientation list; simulate by jumping halfway around
        st.session_state.orient_idx = (st.session_state.orient_idx + len(orients)//2) % len(orients)
        st.rerun()
    if cols[3].button("❌ Deselect", key="deselect"):
        st.session_state.selected_piece = None
        st.rerun()

    # Tiny preview
    sh = orients[st.session_state.orient_idx]
    max_r = max(r for r,_ in sh)
    max_c = max(c for _,c in sh)
    lines = []
    for r in range(max_r+1):
        row = []
        for c in range(max_c+1):
            row.append(PIECE_MAP[pk].color if (r,c) in sh else "▫️")
        lines.append("".join(row))
    st.write("Preview:")
    st.write("\n".join(lines))

def render_solution_grid(solution: Dict[str, Set[Coord]]):
    # Convert to cell->color render
    cell2color = {}
    for pk, cells in solution.items():
        color = PIECE_MAP[pk].color
        for rc in cells:
            cell2color[rc] = color
    # Build HTML grid
    html = ['<div style="display:grid;grid-template-columns: repeat(6, 22px); gap:2px;">']
    for r in range(BOARD_N):
        for c in range(BOARD_N):
            if (r,c) in st.session_state.blockers:
                bg = '#333333'
                txt = '⛔'
                html.append(f'<div style="width:22px;height:22px;display:flex;align-items:center;justify-content:center;font-size:12px">{txt}</div>')
            else:
                sym = cell2color.get((r,c), "▫️")
                html.append(f'<div style="width:22px;height:22px;display:flex;align-items:center;justify-content:center;font-size:12px">{sym}</div>')
    html.append('</div>')
    st.markdown("".join(html), unsafe_allow_html=True)

def reset_everything():
    st.session_state.blockers = set()
    st.session_state.placed = {}
    st.session_state.mode = 'blockers'
    st.session_state.selected_piece = None
    st.session_state.orient_idx = 0
    st.session_state.solutions = []
    st.session_state.page = 1



# -----------------------------
# Similarity utilities (Tab 4)
# -----------------------------
def compute_pairwise_similarity(solutions: List[Dict[str, Set[tuple]]], piece_keys: List[str]) -> List[tuple]:
    """
    Returns list of (i, j, sim) for 1-based solution indices i<j.
    sim = fraction of pieces with identical placed cell sets (position+orientation), normalized by #pieces.
    """
    pairs = []
    M = len(solutions)
    if M < 2:
        return pairs
    denom = max(1, len(piece_keys))
    for i in range(M):
        Si = solutions[i]
        for j in range(i+1, M):
            Sj = solutions[j]
            matches = 0
            for pk in piece_keys:
                cells_i = Si.get(pk)
                cells_j = Sj.get(pk)
                if cells_i is not None and cells_j is not None and cells_i == cells_j:
                    matches += 1
            sim = matches / denom
            pairs.append((i+1, j+1, sim))  # 1-based indices
    return pairs
# -----------------------------
# Streamlit App
# -----------------------------

st.set_page_config(page_title="Genius Square – Streamlit", layout="wide")
st.title("🧠 The Genius Square – Solver & Sandbox")

# Init session state
if "blockers" not in st.session_state:
    reset_everything()

tabs = st.tabs(["🎲 Setup & Play", "🧩 All Solutions", "ℹ️ How to use", "📈 Similarity"])

# --- Tab 1: Setup & Play ---
with tabs[0]:
    left, right = st.columns([2, 1], gap="large")

    with left:
        st.subheader("1) Place the 7 blockers")
        st.caption("Click squares to toggle blockers. Need exactly **7**.")

        st.write(f"**Blockers placed:** {len(st.session_state.blockers)}/7")
        render_grid_interactive(st.session_state.blockers, st.session_state.placed, st.session_state.mode, st.session_state.selected_piece, st.session_state.orient_idx)

        # Mode controls
        st.write("### Mode")
        colm = st.columns(3)
        if colm[0].button("🧱 Blockers mode"):
            st.session_state.mode = 'blockers'; st.rerun()
        if colm[1].button("🎮 Place pieces"):
            st.session_state.mode = 'place'; st.rerun()
        if colm[2].button("🧹 Remove piece"):
            st.session_state.mode = 'remove'; st.rerun()

        st.divider()

        # Action buttons
        bcols = st.columns(3)
        disabled = len(st.session_state.blockers) != 7
        if bcols[0].button("🧑‍🎨 I will solve it", disabled=disabled):
            # Move to play mode and show palette
            st.session_state.mode = 'place'
            st.session_state.selected_piece = None
            st.rerun()
        if bcols[1].button("🤖 Solve it for me", disabled=disabled):
            with st.spinner("Solving… this can take a few seconds for some setups."):
                st.session_state.solutions = solve_all(st.session_state.blockers)
                st.session_state.page = 1
            st.toast(f"Found {len(st.session_state.solutions)} solution(s). See the 'All Solutions' tab.")
            st.rerun()
        if bcols[2].button("🔁 Reset board"):
            reset_everything()
            st.rerun()

    with right:
        st.subheader("2) Piece palette & controls")
        render_piece_palette()
        st.divider()
        render_selected_piece_controls()

# --- Tab 2: All Solutions ---
with tabs[1]:
    st.subheader("All Solutions for current blocker layout")
    if len(st.session_state.blockers) != 7:
        st.info("Place exactly 7 blockers in Tab 1 to enable solving.")
    else:
        if not st.session_state.solutions:
            st.warning("No solutions computed yet. Click **Solve it for me** in Tab 1.")
        else:
            total = len(st.session_state.solutions)
            page_size = 20
            pages = (total + page_size - 1) // page_size
            coltop = st.columns(3)
            st.session_state.page = coltop[0].number_input("Page", min_value=1, max_value=max(1, pages), value=st.session_state.page, step=1)
            coltop[1].markdown(f"**Total solutions:** {total}")
            coltop[2].markdown(f"**Showing:** {page_size} per page")

            start = (st.session_state.page - 1) * page_size
            end = min(total, start + page_size)

            grid_cols = st.columns(4, gap="large")
            slot = 0
            for idx in range(start, end):
                with grid_cols[slot % 4]:
                    st.markdown(f"**#{idx+1}**")
                    render_solution_grid(st.session_state.solutions[idx])
                slot += 1

# --- Tab 3: How to use ---
with tabs[2]:
    st.header("How to move, rotate, and remove pieces")
    st.markdown("""
**Interaction model (touch & mouse friendly):**  

1. **Place blockers**: In *Blockers mode*, click on cells to toggle blockers on/off. You need **exactly 7**.
2. **Solve it yourself**: Click **I will solve it** (or switch to *Place pieces* mode).  
   - Click **🎯 Select** on a piece in the right panel.  
   - Use **⟲ Rotate** / **⟲⟲ Rotate x2** / **⇋ Flip** to change orientation.  
   - Click a cell on the board to **place** the selected piece (the piece's top-left cell anchors to where you click).  
   - To **remove** a piece, switch to *Remove piece* mode and click any square of that piece.
3. **Let the app solve**: Click **Solve it for me**.  
   - Go to the **All Solutions** tab to browse results (**20 per page**).

> Streamlit doesn't provide native drag-and-drop for grid pieces without third‑party components, so this app uses fast **tap-to-place** controls that feel similar in play.  
> If you want true drag-and-drop later, we can swap in a custom component.

**Tips & strategy**  
- Try placing big pieces first (e.g., tetrominoes) and leave flexible pieces (domino & single) for the end.  
- Watch for awkward cavities (1×n corridors or isolated corners) and reserve the appropriate piece.
    """)
    st.caption("Have fun racing! – Made with ❤️ in Streamlit")


# --- Tab 4: Similarity ---
with tabs[3]:
    st.subheader("📈 Solution similarity")
    sols = st.session_state.get("solutions", [])
    # Some versions used 'sols' instead of 'solutions':
    if not sols and "sols" in st.session_state:
        sols = st.session_state.sols
    if not sols:
        st.info("No solutions yet. Click **Solve it for me** in Tab 1.")
    elif len(sols) < 2:
        st.info("Need at least 2 solutions to compute pairwise similarity.")
    else:
        piece_keys = [p.key for p in PIECES]
        pairs = compute_pairwise_similarity(sols, piece_keys)
        if not pairs:
            st.warning("No comparable pairs found.")
        else:
            sims = [p[2] for p in pairs]
            avg = statistics.mean(sims)
            sd = statistics.pstdev(sims) if len(sims) > 1 else 0.0

            # Plot with matplotlib (single plot, no seaborn, no explicit colors)
            import matplotlib.pyplot as plt
            fig = plt.figure()
            xs = list(range(1, len(pairs)+1))
            plt.plot(xs, sims, marker='o', linestyle='none')
            plt.xlabel("Pair index (i<j in enumeration order)")
            plt.ylabel("Similarity (fraction of matching pieces)")
            plt.title(f"Pairwise similarity — mean={avg:.3f}, SD={sd:.3f}")
            st.pyplot(fig)

            # Mapping table under the plot
            import pandas as pd
            df = pd.DataFrame({
                "pair_index": xs,
                "i": [i for (i, j, _) in pairs],
                "j": [j for (i, j, _) in pairs],
                "similarity": sims,
            })
            st.dataframe(df, use_container_width=True)
