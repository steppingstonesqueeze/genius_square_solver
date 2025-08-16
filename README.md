# Genius Square – Streamlit App

A beautiful, self-contained Streamlit app to **play** and **solve** the Genius Square:

- **Tab 1:** Place 7 blockers, then either **solve it yourself** with click‑to‑place, rotate, flip; or click **Solve it for me**.
- **Tab 2:** Browse **all solutions** for the current blockers (20 per page).
- **Tab 3:** Instructions for moving/rotating/removing pieces.

## Run locally

```bash
pip install streamlit
streamlit run app.py
```

No other dependencies required.

## Notes

- The nine pieces implemented are: I, O, T, Z, L tetrominoes; straight triomino; L triomino; domino; single.  
- Rotations and mirror flips are supported.
- If you manually create an unsolvable board (possible when freely placing blockers), the solver will report **0 solutions**.