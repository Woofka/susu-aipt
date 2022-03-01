import time
import numpy as np

from tqdm import tqdm
from apyori import apriori
import matplotlib.pyplot as plt


def load_dataset(filename):
    transactions = []
    print('Loading data... ', end='')
    with open(filename, 'r') as f:
        for line in f.readlines():
            tmp = line.strip().split()
            tmp = list(map(int, tmp))
            transactions.append(tmp)
    print('Done\n')
    return transactions


def print_apriori_results(results, sort_by_support=True):
    if sort_by_support:
        results = sorted(results, key=lambda x: x.support, reverse=True)
    else:
        results = sorted(results, key=lambda x: x.items)

    for r in results:
        print(f'Items: {list(r.items)}  Support: {r.support:.2f}')
        for stat in r.ordered_statistics:
            base = None if len(stat.items_base) == 0 else list(stat.items_base)
            print(f'    {base} -> {list(stat.items_add)}  Confidence: {stat.confidence:.2f}  Lift: {stat.lift:.2f}')


def single_run():
    transactions = load_dataset('retail.dat')

    """
    min_support -- The minimum support of relations (float). Default = 0.1
    min_confidence -- The minimum confidence of relations (float). Default = 0.0
    min_lift -- The minimum lift of relations (float). Default = 0.0
    max_length -- The maximum length of the relation (integer). Default = None
    """
    results = list(apriori(transactions, min_support=0.03, min_confidence=0.0, min_lift=0.0, max_length=3))
    print_apriori_results(results, sort_by_support=True)


def analyse_time_per_supp():
    transactions = load_dataset('retail.dat')
    supports = {}
    print('Analysing: dependence of time on support')
    for _ in tqdm(range(25)):
        for supp in np.linspace(0, 1, 41):
            if supp == 0:
                continue
            supp = round(supp, 3)
            if supp not in supports:
                supports[supp] = []

            t = time.time()
            list(apriori(transactions, min_support=supp, max_length=3))
            t = time.time() - t
            supports[supp].append(t)
    resx, resy = [], []
    for supp, times in supports.items():
        resx.append(supp)
        resy.append(sum(times)/len(times))

    plt.plot(resx, resy)
    plt.title('Dependence of time on support')
    plt.xlabel('Support')
    plt.ylabel('Time (s)')
    plt.grid()
    plt.savefig('time_per_supp.png')


def analyse_amount_per_supp():
    transactions = load_dataset('retail.dat')
    max_len = 3
    supports = {}

    print('Analysing: dependence of sets amount on support')
    for supp in tqdm(np.linspace(0, 1, 41)):
        if supp == 0:
            continue
        supp = round(supp, 3)

        amounts = {}
        for i in range(1, max_len + 1):
            amounts[i] = 0

        results = list(apriori(transactions, min_support=supp, max_length=max_len))

        for l in range(1, max_len+1):
            amounts[l] += len(list(filter(lambda x: len(x.items) == l, results)))

        supports[supp] = amounts

    resx = []
    resy = [[] for _ in range(max_len)]
    for supp, amounts in supports.items():
        resx.append(supp)
        for size, amount in amounts.items():
            resy[size-1].append(amount)

    fig, ax = plt.subplots()
    for i, y in enumerate(resy):
        ax.plot(resx, y, label=f'Len={i+1}')
    ax.legend()
    plt.title('Dependence of sets amount on support')
    plt.xlabel('Support')
    plt.ylabel('Sets amount')
    plt.grid()
    plt.savefig('amount_per_supp.png')


def main():
    single_run()
    # analyse_time_per_supp()
    # analyse_amount_per_supp()


if __name__ == '__main__':
    main()
