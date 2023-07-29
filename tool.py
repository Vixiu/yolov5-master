from math import pi, tan, atan

# https://www.mouse-sensitivity.com/







class FOV:
    def __init__(self, game_width, game_fov, game_pixel, degrees, measure_seen, game_seen) -> None:
        '''
        :param game_width: 游戏分辨率宽度
        :param game_fov: 游戏的HFov

        :param game_pixel: 游戏的一圈花了多少像素
        :param degrees: 游戏一圈的360°
        :param measure_seen: 测量像素时的灵敏度
        :param game_seen: 游戏的灵敏度
        '''
        self.game_width = game_width
        self.game_fov = game_fov
        self.game_pixel = game_pixel
        self.degrees = degrees
        self.measure_seen = measure_seen
        self.game_seen = game_seen

    def __call__(self, target_move) -> float:
        x = (self.game_width / 2) / (tan((self.game_fov * pi / 180) / 2))
        return (
            (atan(target_move / x))
            * (self.game_pixel / (self.degrees * pi / 180))
            * (self.measure_seen / self.game_seen)
        )