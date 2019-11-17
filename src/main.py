import argparse

from .tvp import train, plot_lca

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('some')

    args = parser.parse_args()

    train()
