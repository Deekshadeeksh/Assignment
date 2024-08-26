import numpy as np
import pygame
import random


pygame.init()


GRID_SIZE = 4
TILE_SIZE = 100
MARGIN = 10
WINDOW_SIZE = GRID_SIZE * TILE_SIZE + (GRID_SIZE + 1) * MARGIN + 100
BACKGROUND_COLOR = (187, 173, 160)
TILE_COLORS = {
    0: (204, 192, 179),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
}
FONT_COLOR = (119, 110, 101)
SCORE_COLOR = (255, 255, 255)


screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption('2048')


def init_grid():
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    add_random_tile(grid)
    add_random_tile(grid)
    return grid


def add_random_tile(grid):
    empty_cells = list(zip(*np.where(grid == 0)))
    if empty_cells:
        i, j = random.choice(empty_cells)
        grid[i][j] = 2 if random.random() < 0.9 else 4


def draw_grid(grid, score):
    screen.fill(BACKGROUND_COLOR)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            value = grid[i][j]
            color = TILE_COLORS.get(value, TILE_COLORS[2048])
            rect = pygame.Rect(
                j * TILE_SIZE + (j + 1) * MARGIN,
                i * TILE_SIZE + (i + 1) * MARGIN + 100,
                TILE_SIZE, TILE_SIZE
            )
            pygame.draw.rect(screen, color, rect)
            if value != 0:
                font = pygame.font.SysFont(None, 55)
                text_surface = font.render(f"{value}", True, FONT_COLOR)
                text_rect = text_surface.get_rect(center=rect.center)
                screen.blit(text_surface, text_rect)


    font = pygame.font.SysFont(None, 50)
    score_surface = font.render(f"Score: {score}", True, SCORE_COLOR)
    screen.blit(score_surface, (MARGIN, MARGIN))

    pygame.display.flip()


def compress(grid):
    new_grid = np.zeros_like(grid)
    for i in range(GRID_SIZE):
        position = 0
        for j in range(GRID_SIZE):
            if grid[i][j] != 0:
                new_grid[i][position] = grid[i][j]
                position += 1
    return new_grid


def merge(grid, score):
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE - 1):
            if grid[i][j] == grid[i][j + 1] and grid[i][j] != 0:
                grid[i][j] *= 2
                score += grid[i][j]
                grid[i][j + 1] = 0

    return grid, score


def move_left(grid, score):
    new_grid = compress(grid)
    new_grid, score = merge(new_grid, score)
    new_grid = compress(new_grid)
    return new_grid, score


def move_right(grid, score):
    reversed_grid = np.fliplr(grid)
    new_grid, score = move_left(reversed_grid, score)
    return np.fliplr(new_grid), score


def move_up(grid, score):
    transposed_grid = np.transpose(grid)
    new_grid, score = move_left(transposed_grid, score)
    return np.transpose(new_grid), score


def move_down(grid, score):
    transposed_grid = np.transpose(grid)
    new_grid, score = move_right(transposed_grid, score)
    return np.transpose(new_grid), score


def expectimax(board, depth, is_max_turn):
    if depth == 0 or game_over(board):
        return evaluate_board(board)

    if is_max_turn:
        best_score = float('-inf')
        for move in get_all_possible_moves(board):
            score = expectimax(simulate_move(board, move), depth - 1, False)
            best_score = max(best_score, score)
        return best_score
    else:
        scores = []
        for new_board in get_all_possible_new_boards(board):
            score = expectimax(new_board, depth - 1, True)
            scores.append(score)
        return sum(scores) / len(scores)


def select_best_move(board):
    best_move = None
    best_score = float('-inf')
    for move in get_all_possible_moves(board):
        score = expectimax(simulate_move(board, move), depth=3, is_max_turn=False)
        if score > best_score:
            best_score = score
            best_move = move
    return best_move


def get_all_possible_moves(board):
    return ['UP', 'DOWN', 'LEFT', 'RIGHT']


def simulate_move(board, move):
    if move == 'UP':
        return move_up(board.copy(), 0)[0]
    elif move == 'DOWN':
        return move_down(board.copy(), 0)[0]
    elif move == 'LEFT':
        return move_left(board.copy(), 0)[0]
    elif move == 'RIGHT':
        return move_right(board.copy(), 0)[0]


def get_all_possible_new_boards(board):
    new_boards = []
    empty_cells = list(zip(*np.where(board == 0)))
    for i, j in empty_cells:
        for value in [2, 4]:
            new_board = board.copy()
            new_board[i][j] = value
            new_boards.append(new_board)
    return new_boards


def evaluate_board(board):
    return np.sum(board)


def game_over(board):
    for move in get_all_possible_moves(board):
        if not np.array_equal(board, simulate_move(board, move)):
            return False
    return True


def main():
    grid = init_grid()
    score = 0
    clock = pygame.time.Clock()

    while True:
        draw_grid(grid, score)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return


        move = select_best_move(grid)
        if move == 'UP':
            grid, score = move_up(grid, score)
        elif move == 'DOWN':
            grid, score = move_down(grid, score)
        elif move == 'LEFT':
            grid, score = move_left(grid, score)
        elif move == 'RIGHT':
            grid, score = move_right(grid, score)

        add_random_tile(grid)

        if game_over(grid):
            print(f"Game Over! Final Score: {score}")
            pygame.quit()
            return

        clock.tick(10)


if __name__ == "__main__":
    main()
