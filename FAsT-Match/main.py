from MatchNet import MatchNet


def fast_match():
    # just some functionality testing
    net = MatchNet(1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1)
    print(net)
    print(net * 3)
    print(net / 5)


if __name__ == '__main__':
    fast_match()
