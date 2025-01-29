import multiprocessing
import time

SIZE = 8

# DO NOT CHANGE THE ORDER OF THE ITEMS!
DIRECTIONS = [(1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)]
DIRECTIONS_OFFSETS = [SIZE + 1, SIZE, SIZE - 1, -1, -(SIZE + 1), -SIZE, -(SIZE - 1), 1]


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


def apply_move(board: tuple[int, int], move_flip: tuple[int, int]) -> tuple[int, int]:
    taken_places, colors = board
    position_flip, colors_flip = move_flip

    return taken_places ^ position_flip, colors ^ colors_flip


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


def translate_bin_board_to_01none_lists(board: tuple[int, int]) -> list[list[int | None]]:
    taken_places, colors = board

    rows = []
    bit = 1
    for i in range(SIZE):
        row = []
        for j in range(SIZE):
            is_taken_place = taken_places & bit

            if is_taken_place:
                color = (colors & bit) > 0
                row.append(color)
            else:
                row.append(None)
            bit <<= 1

        rows.append(row)
    return rows


def translate_bin_move_to_row_col(bin_move):
    move_position_bit, _ = bin_move
    position_in_board = 0
    while move_position_bit:
        position_in_board += 1
        move_position_bit >>= 1
    position_in_board -= 1

    row, col = divmod(position_in_board, SIZE)
    return row, col


def count_binary_1s(n: int) -> int:
    c = 0
    while n != 0:
        if n & 1:
            c += 1
        n >>= 1
    return c


def preview_board(board: tuple[int, int]):
    taken_places, colors = board

    bit = 1
    board_repr = ""
    for i in range(SIZE):
        for j in range(SIZE):
            is_place_taken = taken_places & bit
            if not is_place_taken:
                board_repr += "."
            else:
                board_repr += "1" if colors & bit else "0"
            bit <<= 1
        board_repr += "\n"
    return board_repr


STARTING_BOARD = [[0] * SIZE for _ in range(SIZE)]
STARTING_BOARD[SIZE // 2 - 1][SIZE // 2 - 1] = 1
STARTING_BOARD[SIZE // 2][SIZE // 2] = 1
STARTING_BOARD[SIZE // 2 - 1][SIZE // 2] = 2
STARTING_BOARD[SIZE // 2][SIZE // 2 - 1] = 2

STARTING_BIN_BOARD = translate_to_bin(STARTING_BOARD)


########################################################################################################################


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


def order_moves(moves, start_with_move):
    moves__with_values = []
    for move in moves:
        if move == start_with_move:
            value = float('inf')

        else:
            location, color_flip = move
            pieces_flipped = count_binary_1s(color_flip)
            if location & color_flip == location:
                pieces_flipped -= 1
            value = -pieces_flipped

            row, col = translate_bin_move_to_row_col(move)
            if row == 0 or row == SIZE - 1:
                value += 7
            if col == 0 or col == SIZE - 1:
                value += 7

        moves__with_values.append([value, move])

    ordered_moves = sorted(moves__with_values, key=lambda x: x[0])
    return [x[1] for x in ordered_moves]


def rotate_masking(masking: list[list]):
    result = []
    for row in masking:
        result.append(row + row[::-1])

    n = len(result)
    for i in range(n):
        result.append(result[n - i - 1])
    return result


MASKING = [
    [100, -15, 15, 5],
    [-15, -50, 2, 0],
    [15, 2, 15, 0],
    [5, 0, 0, 3]
]

MASKING = rotate_masking(MASKING)


def evaluate_board(board):
    bit = 1
    taken_places, colors = board
    first_player_taken_places, second_player_taken_places = 0, 0

    value = 0
    for row in range(SIZE):
        for col in range(SIZE):
            if taken_places & bit:
                masking_value = MASKING[row][col]

                if colors & bit:
                    value += masking_value
                    first_player_taken_places += 1
                else:
                    value -= masking_value
                    second_player_taken_places += 1

            bit <<= 1
    number_of_taken_places = first_player_taken_places + second_player_taken_places

    if number_of_taken_places >= 55:
        value += 10 * (first_player_taken_places - second_player_taken_places)
    else:
        value -= 0.1 * number_of_taken_places * (first_player_taken_places - second_player_taken_places)

    if first_player_taken_places == 0:
        return float('-inf')
    elif second_player_taken_places == 0:
        return float('inf')

    first_player_options_number = len(get_available_moves(1, board))
    second_player_options_number = len(get_available_moves(0, board))
    value += number_of_taken_places / 4 * (first_player_options_number - second_player_options_number)

    lists_board = translate_bin_board_to_01none_lists(board)
    strips = []
    for i in range(SIZE):
        row_strip = []
        col_strip = []
        for j in range(SIZE):
            row_strip.append(lists_board[i][j])
            col_strip.append(lists_board[j][i])
        # maybe check if the strip is full
        strips.append(row_strip)
        strips.append(col_strip)

    for i in range(3, SIZE):
        left_diagonal_strip = []  # like /
        right_diagonal_strip = []  # like \
        for j in range(i + 1):
            left_diagonal_strip.append(lists_board[j][i - j])
            right_diagonal_strip.append(lists_board[j][(SIZE - i - 1) + j])
        # maybe check if the strip is full
        strips.append(left_diagonal_strip)
        strips.append(right_diagonal_strip)

    for strip in strips:
        sandwich_bread = None
        sandwich_content = None
        bread_start_index = 0
        content_start_index = 0

        for i in range(len(strip)):
            color = strip[i]

            if color is None:
                sandwich_bread = None
                sandwich_content = None
                continue

            if sandwich_bread is None:
                sandwich_bread = color
                bread_start_index = i

            else:
                if color != sandwich_bread:
                    if sandwich_content is None:
                        sandwich_content = color
                        content_start_index = i

                elif sandwich_content is not None:  # a sandwich is ready
                    # score sandwich: TODO
                    if bread_start_index == 0 or i == SIZE - 1:
                        pass
                    else:
                        pass

                    sandwich_bread = sandwich_content
                    bread_start_index = content_start_index
                    content_start_index = i

    return value


########################################################################################################################

best_move_of_last_search = None
current_search_best_move = None


def minimax(board, depth, maximizing_player: bool, distance_from_root=0, alpha=-float('inf'), beta=float('inf'),
            had_moves_last_turn=True):
    global current_search_best_move

    available_moves = get_available_moves(1 if maximizing_player else 0, board)
    if distance_from_root == 0:
        available_moves = order_moves(available_moves, best_move_of_last_search)
    else:
        available_moves = order_moves(available_moves, None)

    if len(available_moves) == 0:
        if not had_moves_last_turn:
            return None, evaluate_board(board)
        return minimax(board, depth, not maximizing_player, distance_from_root + 1, alpha, beta, False)

    if maximizing_player:
        best_value = -float('inf')
        best_move = None

        for move in available_moves:
            board = apply_move(board, move)
            if depth == 1:
                child_value = evaluate_board(board)
            else:
                _, child_value, = minimax(board, depth - 1, False, distance_from_root + 1, alpha, beta)

            board = apply_move(board, move)  # reverse board to previous state

            if child_value > best_value:
                best_value = child_value
                best_move = move
                if distance_from_root == 0:
                    current_search_best_move = best_move

            alpha = max(alpha, best_value)
            if beta <= alpha:
                break

    else:
        best_value = float('inf')
        best_move = None
        for move in available_moves:
            board = apply_move(board, move)
            if depth == 1:
                child_value = evaluate_board(board)
            else:
                _, child_value = minimax(board, depth - 1, True, distance_from_root + 1, alpha, beta)
            board = apply_move(board, move)  # reverse board to previous state

            if child_value < best_value:
                best_value = child_value
                best_move = move
                if distance_from_root == 0:
                    current_search_best_move = best_move

            beta = min(beta, best_value)
            if beta <= alpha:
                break

    return best_move, best_value


def iterative_deepening_minimax(board, me, max_time, return_dict):
    start_time = time.time()

    global best_move_of_last_search

    depth = 1
    while True:
        best_move, _ = minimax(board, depth, me)

        if time.time() - start_time < max_time:
            return_dict['best_move'] = best_move
            best_move_of_last_search = best_move
        else:
            break

        print(depth)
        depth += 1


########################################################################################################################
def get_move(me: int, board: list[list[int]]) -> tuple[int, int]:
    bin_board = translate_to_bin(board)
    me = me - 1

    # Set the maximum time limit for minimax
    max_time = 0.48  # 0.6 seconds

    # Initialize a return dictionary to collect the best move
    return_dict = multiprocessing.Manager().dict()

    # Create a process to run minimax with a time limit
    process = multiprocessing.Process(target=iterative_deepening_minimax, args=(bin_board, me, max_time, return_dict))
    process.start()
    process.join(timeout=max_time)

    # If the process is still running, terminate it
    if process.is_alive():
        process.terminate()
        process.join()

    # Get the best move from the return dictionary
    best_move = return_dict.get('best_move', None)

    if best_move is None:
        best_move = current_search_best_move

    return translate_bin_move_to_row_col(best_move)


# Nice function for playing against ourselves
# Sandwich maybe with bits
# Add evaluation for stable disks that cannot be flipped
# When there is stable for example a corner or an e
# Density function


def main(*_):
    # t1 = time.perf_counter()
    # minimax(STARTING_BIN_BOARD, 9, True)
    # t2 = time.perf_counter()
    # print(t2 - t1)

    return_dict = multiprocessing.Manager().dict()
    process = multiprocessing.Process(target=iterative_deepening_minimax,
                                      args=(STARTING_BIN_BOARD, 1, 10, return_dict))

    process.start()
    process.join(timeout=10)
    if process.is_alive():
        process.terminate()
        process.join()
    # iterative_deepening_minimax(STARTING_BIN_BOARD, 0.5, 1, return_dict)

    print(return_dict.get('best_move', None))


if __name__ == '__main__':
    main()