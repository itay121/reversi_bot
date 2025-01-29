# reversi_bot

This is an AI bot developed for a Reversi competition. The bot achieved **first place** with a record of **14 wins, 0 draws, and 0 losses**.

---

## Core Features:

### Bitwise Board Representation
- The board is represented using two 64-bit integers:
  - `taken_places` (occupied positions)
  - `colors` (which positions belong to each player)
- Efficient storage and operations using bitwise operators like `|`, `&`, `^`, and shifts (`<<`, `>>`).

---

### Move Calculation and Application
- **`calculate_move`**: Computes the effect of a move, flipping the opponent's pieces as necessary.
- **`apply_move`**: Updates the board state after applying the move.

---

### Move Validation
- **`get_available_moves`**: Identifies all valid moves for a player based on the current board state.
- Ensures move validity by checking that a chain is formed between the player's piece and the current move.

---

### Heuristic Evaluation
- **`evaluate_board`**: Provides a score for the current board state based on several factors:
  - **Position Masking**: Uses a predefined positional score matrix (`MASKING`) to prioritize certain areas.
  - **Density**: Evaluates local control using a "density mask" that considers neighboring pieces.
  - **Endgame Considerations**: Adds weight to moves during the late game (55+ pieces on the board).
  - **Mobility**: Rewards players with more available moves.
  - **Sandwich Heuristic** (incomplete): Checks for sequences of opponent pieces enclosed by the player’s pieces.

---

### Move Ordering
- **`order_moves`**: Prioritizes moves based on potential piece flips, corner captures, and other positional advantages.

---

### Neighbor Count
- **`count_neighbors`**: Calculates the number of neighbors for each position on the board.
- Useful for evaluating density and potential control areas.

---

### Minimax Algorithm (Partial Implementation)
- The beginnings of a **minimax** implementation with **alpha-beta pruning** to explore the game tree.
- Uses **`evaluate_board`** as the heuristic function for leaf nodes.

---

### Utilities
- **Board Translation**:
  - Converts between binary and list representations (`translate_to_bin`, `translate_bin_board_to_01none_lists`).
- **Visualization**:
  - **`preview_board`**: Generates a human-readable string representation of the board.
- **Rotations**:
  - Rotates matrices (e.g., for positional evaluation masks).

---

### Game-Specific Heuristics
- **Density**:
  - Combines board state and neighbor information for positional evaluation.
- **Sandwich Detection** (partially implemented):
  - Will add scoring for capturing enclosed lines of pieces.
