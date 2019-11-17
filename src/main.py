import argparse

from .tvp import train, plot_grad

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('some')

    args = parser.parse_args()

    lca_val = train()

    # check lca is
    lca_sum = 0
    for k, v in lca_val.items():
        print(v.sum())
        lca_sum += v.sum().item()
    print(lca_sum)

    # plot
    plot_grad(lca_val)
