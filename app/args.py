from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default='ag_news', help='Dataset name')
parser.add_argument('-i','--iteration', type=int, default=30, help='Number of iterations')
args = parser.parse_args()