from typing import Callable

from alexnet import main as alexnet_main
from googlenet import main as googlenet_main
from lenet import main as lenet_main
from nin import main as nin_main
from vgg import main as vgg_main
from resnet import main as resnet_main
from densenet import main as densenet_main


def main():
    cases: list[tuple[str, Callable[[],]]] = []
    cases.append(('lenet', lenet_main))
    cases.append(('alexnet', alexnet_main))
    cases.append(('vgg', vgg_main))
    cases.append(('nin', nin_main))
    cases.append(('googlenet', googlenet_main))
    cases.append(('resnet', resnet_main))
    cases.append(('densenet', densenet_main))
    for case, main in cases:
        print(f'build {case}')
        main()


if __name__ == '__main__':
    main()