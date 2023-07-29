def getMonitor(screenW, screenH, Width, Height) -> dict:
    """
    返回monitor字典
    """

    LeftTopX, LeftTopY = int(screenW / 2 - Width / 2), int(screenH / 2 - Height / 2)
    return {'left': LeftTopX, 'top': LeftTopY, 'width': Width, 'height': Height}


def a(q, w,e,r):
    print(q, w,e,r)


def b():
    return 1,2


a(*b()*2)
