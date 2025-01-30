# Most of this functions are also included in the bot file but I wanted that comparing bots would work which other bots that has get_move() function

def board_to_fen(board):
    piece_map = {1: 'p', 2: 'P', 0: None}  # Black pawn for 1, White pawn for 2
    fen_rows = []

    for row in board:
        fen_row = ""
        empty_count = 0

        for cell in row:
            if cell == 0:
                empty_count += 1
            else:
                if empty_count:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += piece_map[cell]

        if empty_count:
            fen_row += str(empty_count)

        fen_rows.append(fen_row)

    return "/".join(fen_rows) + " w - - 0 1"  # Adding chess-specific FEN fields


def move_to_square_notation(move):
    return f"{chr(move[1] + 97)}{8 - move[0]}"


def print_result(board):
    black_amount = sum(row.count(1) for row in board)
    white_amount = sum(row.count(2) for row in board)

    if black_amount > white_amount:
        print('Black won ' + str(black_amount) + '-' + str(white_amount))
    elif white_amount > black_amount:
        print('White won ' + str(white_amount) + '-' + str(black_amount))
    else:
        print('Draw! ' + str(white_amount) + '-' + str(black_amount))



def transform_reversi_board(board):
    places, colors = board
    board = [[0] * 8 for _ in range(8)]
    
    for i in range(64):
        if (places >> i) & 1:  # Check if there's a piece at position i
            board[i // 8][i % 8] = 1 if (colors >> i) & 1 else 2  # Black = 1, White = 2
    
    return board


def is_game_over(board):
    bin_board = translate_to_bin(board)
    if len(get_available_moves(0, bin_board)) == 0 and len(get_available_moves(1, bin_board)) == 0:
        return True
    else:
        return False

def get_available_moves(me, board):
    moves = []

    taken_places, colors = board

    bit = 1
    for row in range(SIZE):
        for col in range(SIZE):
            if not taken_places & bit:  # I.E. place is not taken
                move = (row, col)
                bin_move = calculate_move(me, move, board)
                position_bit, colors_flip = bin_move
                if (colors_flip & (~position_bit)) != 0:  # equivalent to "is valid move?"
                    moves.append(bin_move)

            bit <<= 1
    return moves


def translate_to_bin(board: list[list[int]]) -> tuple[int, int]:
    """
    2 values -> 1
    1 values -> 0
    | or , & and , ^ xor , ~ not , << shift left, >> shift right
    """
    taken_places, colors = 0, 0
    bit = 1

    for row in range(SIZE):
        for col in range(SIZE):
            value = board[row][col]
            if value != 0:
                taken_places |= bit

            if value == 2:
                colors |= bit

            bit <<= 1

    return taken_places, colors


def apply_move(board: tuple[int, int], move_flip: tuple[int, int]) -> tuple[int, int]:
    taken_places, colors = board
    position_flip, colors_flip = move_flip

    return taken_places ^ position_flip, colors ^ colors_flip

def calculate_move(me: int, move: tuple[int, int], board: tuple[int, int]) -> tuple[int, int]:
    """
    `me` should be 0 or 1.
    """
    taken_places, colors = board

    row, col = move
    move_position_bit = 1 << (row * SIZE + col)

    colors_flip = 0

    for (d_row, d_col), bit_offset in zip(DIRECTIONS, DIRECTIONS_OFFSETS):
        row, col = move

        bit = move_position_bit
        flip_n = 0

        should_affect = False

        while True:
            row += d_row
            col += d_col

            if row >= SIZE or row < 0 or col >= SIZE or col < 0:
                break

            if bit_offset < 0:
                bit >>= -bit_offset
            else:
                bit <<= bit_offset

            if not (taken_places & bit):  # encountered empty place
                break

            if ((colors & bit) > 0) is (me > 0):  # a connection was made
                should_affect = True
                break

            flip_n |= bit

        if should_affect:
            colors_flip ^= flip_n

    if me:
        colors_flip |= move_position_bit  # flip the selected move position color

    return move_position_bit, colors_flip


SIZE = 8

DIRECTIONS = [(1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)]
DIRECTIONS_OFFSETS = [SIZE + 1, SIZE, SIZE - 1, -1, -(SIZE + 1), -SIZE, -(SIZE - 1), 1]

STARTING_BOARD = [[0] * SIZE for _ in range(SIZE)]
STARTING_BOARD[SIZE // 2 - 1][SIZE // 2 - 1] = 1
STARTING_BOARD[SIZE // 2][SIZE // 2] = 1
STARTING_BOARD[SIZE // 2 - 1][SIZE // 2] = 2
STARTING_BOARD[SIZE // 2][SIZE // 2 - 1] = 2

STARTING_BIN_BOARD = translate_to_bin(STARTING_BOARD)
