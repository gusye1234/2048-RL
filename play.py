from enum import Flag
import pygame
import sys
import os
import sys
import torch
sys.path.append("./code")
import numpy as np
from game2048 import Game2048
from network import *
import math

'''FPS'''
FPS = 60

BG_COLOR = '#92877d'

SCREENSIZE = (370, 450)

'''字体路径'''
FONTPATH = os.path.join(os.getcwd(), './resources/JetBrainsMono-Bold.ttf')
'''背景音乐路径'''
BGMPATH = os.path.join(os.getcwd(), './resources/bgm.mp3')
MODELPATH = "./resources/best_512.pkl"
'''其他一些必要的常量'''
MARGIN_SIZE = 10
BLOCK_SIZE = 80
GAME_MATRIX_SIZE = (4, 4)
# GAME_MATRIX_SIZE = (5, 5)
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
NETWORK = Network().to(DEVICE)
NETWORK.load_state_dict(torch.load(MODELPATH, map_location=DEVICE))

'''根据方格当前的分数获得[方格背景颜色, 方格里的字体颜色]'''
def getColorByNumber(number):
    number2color_dict = {
                            2: ['#eee4da', '#776e65'], 4: ['#ede0c8', '#776e65'], 8: ['#f2b179', '#f9f6f2'],
                            16: ['#f59563', '#f9f6f2'], 32: ['#f67c5f', '#f9f6f2'], 64: ['#f65e3b', '#f9f6f2'],
                            128: ['#edcf72', '#f9f6f2'], 256: ['#edcc61', '#f9f6f2'], 512: ['#edc850', '#f9f6f2'],
                            1024: ['#edc53f', '#f9f6f2'], 2048: ['#edc22e', '#f9f6f2'], 4096: ['#eee4da', '#776e65'],
                            8192: ['#edc22e', '#f9f6f2'], 16384: ['#f2b179', '#776e65'], 32768: ['#f59563', '#776e65'],
                            65536: ['#f67c5f', '#f9f6f2'], 0: ['#9e948a', None]
                        }
    return number2color_dict[number]


'''将2048游戏的当前数字排列画到屏幕上'''
def drawGameMatrix(screen, game_matrix):
    screen.fill(pygame.Color(BG_COLOR))
    for i in range(len(game_matrix)):
        for j in range(len(game_matrix[i])):
            number = game_matrix[i][j]
            x = MARGIN_SIZE * (j + 1) + BLOCK_SIZE * j
            y = MARGIN_SIZE * (i + 1) + BLOCK_SIZE * i
            pygame.draw.rect(screen, pygame.Color(getColorByNumber(number)[0]), (x, y, BLOCK_SIZE, BLOCK_SIZE))
            if number != 0:
                font_color = pygame.Color(getColorByNumber(number)[1])
                font_size = BLOCK_SIZE - 15 * len(str(number))
                font = pygame.font.Font(FONTPATH, font_size)
                text = font.render(str(number), True, font_color)
                text_rect = text.get_rect()
                text_rect.centerx, text_rect.centery = x + BLOCK_SIZE / 2, y + BLOCK_SIZE / 2
                screen.blit(text, text_rect)


'''将游戏的最高分和当前分数画到屏幕上'''
def drawScore(screen, score, max_score=2048):
    font_color = (255, 255, 255)
    font_size = 70
    font = pygame.font.Font(FONTPATH, font_size)
    # text_max_score = font.render('Best: %s' % max_score, True, font_color)
    text_score = font.render('%s' % score, True, font_color)
    # start_x = BLOCK_SIZE * GAME_MATRIX_SIZE[1] + MARGIN_SIZE * (GAME_MATRIX_SIZE[1] + 1)
    start_x = 10
    start_y = BLOCK_SIZE * \
        GAME_MATRIX_SIZE[1] + MARGIN_SIZE * (GAME_MATRIX_SIZE[1] + 1)
    # screen.blit(text_max_score, (start_x+10, 10))
    screen.blit(text_score, (start_x, start_y))
    start_y = 30 + text_score.get_rect().height
    return (start_x, start_y)

'''游戏结束界面'''
def endInterface(screen, score):
    font_size_big = 60
    font_size_small = 20
    font_color = (128, 128, 128)
    font_big = pygame.font.Font(FONTPATH, font_size_big)
    font_small = pygame.font.Font(FONTPATH, font_size_small)
    surface = screen.convert_alpha()
    surface.fill((127, 255, 212, 2))
    text = font_big.render(f'Score {score}', True, font_color)
    text_rect = text.get_rect()
    text_rect.centerx, text_rect.centery = SCREENSIZE[0]/2, SCREENSIZE[1]/2-50
    surface.blit(text, text_rect)
    button_width, button_height = 100, 40
    button_start_x_left = SCREENSIZE[0] / 2 - button_width - 20
    button_start_x_right = SCREENSIZE[0] / 2 + 20
    button_start_y = SCREENSIZE[1] / 2 - button_height / 2 + 20
    pygame.draw.rect(surface, (0, 255, 255), (button_start_x_left, button_start_y, button_width, button_height))
    text_restart = font_small.render('Restart', True, font_color)
    text_restart_rect = text_restart.get_rect()
    text_restart_rect.centerx, text_restart_rect.centery = button_start_x_left + button_width / 2, button_start_y + button_height / 2
    surface.blit(text_restart, text_restart_rect)
    pygame.draw.rect(surface, (0, 255, 255), (button_start_x_right, button_start_y, button_width, button_height))
    text_quit = font_small.render('Quit', True, font_color)
    text_quit_rect = text_quit.get_rect()
    text_quit_rect.centerx, text_quit_rect.centery = button_start_x_right + button_width / 2, button_start_y + button_height / 2
    surface.blit(text_quit, text_quit_rect)
    while True:
        screen.blit(surface, (0, 0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button:
                if text_quit_rect.collidepoint(pygame.mouse.get_pos()):
                    return False
                if text_restart_rect.collidepoint(pygame.mouse.get_pos()):
                    return True
        pygame.display.update()

def is_rect(pos,rect):
    x,y =pos
    rx,ry,rw,rh = rect
    if (rx <= x <=rx+rw) and (ry <= y <= ry +rh):
        return True
    return False

def transform_state(state):
    assert type(state) in (tuple,np.ndarray)
    def transform_one(state):
        new = np.zeros(shape=(1,16,4,4),dtype=np.float32)
        for i in range(4):
            for j in range(4):
                if(state[i][j]==0):
                    new[0][0][i][j] = 1.0
                else:
                    loc = int(math.log(state[i][j],2))
                    new[0][loc][i][j] = 1.0
        return new
    if type(state) is np.ndarray:
        return transform_one(state)
    else:
        return np.concatenate([transform_one(s) for s in state],axis=0)

def predict(state):
    state_torch = torch.from_numpy(transform_state(state)).to(DEVICE)
    result = NETWORK.forward(state_torch)
    actionScore = result.cpu().detach().numpy()[0]
    legals = Game2048.legal_moves(state)
    for action in np.argsort(-actionScore):
        if action in legals:
            return action


'''主程序'''
def main():
    # 游戏初始化
    pygame.init()
    screen = pygame.display.set_mode(SCREENSIZE)
    pygame.display.set_caption('2048')
    # 播放背景音乐
    pygame.mixer.music.load(BGMPATH)
    pygame.mixer.music.play(-1)
    # 实例化2048游戏
    game = Game2048()
    # 游戏主循环
    clock = pygame.time.Clock()
    is_running = True
    automate = False
    while is_running:
        screen.fill(pygame.Color(BG_COLOR))
        # --按键检测
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT] and (not automate):
                    ismove,movescore,state = game.move(game.matrix,{pygame.K_UP: 'w', pygame.K_DOWN: 's', pygame.K_LEFT: 'a', pygame.K_RIGHT: 'd'}[event.key])
                    game.score += movescore
                    if ismove:
                        game.generate()
                elif event.key == pygame.K_RETURN:
                    action = predict(game.matrix.copy())
                    ismove, movescore, nextstate = game.move(
                        game.matrix, action)
                    game.matrix = nextstate
                    game.score += movescore
                    if ismove:
                        game.generate()
                if event.key == pygame.K_a:
                    while not game.isover:
                        action = predict(game.matrix.copy())
                        ismove, movescore, nextstate = game.move(
                            game.matrix, action)
                        game.matrix = nextstate
                        game.score += movescore
                        if ismove:
                            game.generate()
                        drawGameMatrix(screen, game.matrix)
                        # drawScore(screen, game.score)
                        pygame.display.update()
                        clock.tick(FPS)
            if game.isover:
                is_running = False
                break
        
        drawGameMatrix(screen, game.matrix)
        drawScore(screen, game.score)
        # drawScore(screen, game.score,max_score)
        # drawGameIntro(screen, start_x, start_y, cfg)
        pygame.display.update()
        clock.tick(FPS)
  
    return endInterface(screen, game.score)


'''run'''
if __name__ == '__main__':
    while True:
        if not main():
            break