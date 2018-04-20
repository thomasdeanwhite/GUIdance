

from yolo import Yolo


if __name__ == '__main__':
    yolo = Yolo()

    yolo.create_network()

    print(yolo.get_network().shape)
