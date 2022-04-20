import time
import numpy as np

import prettytable
from tqdm import tqdm
from apyori import apriori
import matplotlib.pyplot as plt


def load_dataset(filename, sep):
    transactions = []
    print('Loading data... ', end='')
    with open(filename, 'r') as f:
        for line in f.readlines():
            tmp = line.strip().split(sep=sep)
            transactions.append(tmp)
    print('Done\n')
    return transactions


def print_apriori_results(results, sort_by=None):
    table = prettytable.PrettyTable(field_names=['Items', 'Support'])

    for r in results:
        table.add_row([", ".join(list(r.items)), round(r.support, 2)])

    table.align = 'l'
    if sort_by is None:
        print(table)
    elif sort_by == 'support':
        print(table.get_string(sortby='Support', reversesort=True))
    else:
        print(table.get_string(sort_by='Items'))


def print_apriori_results_rules(results, sort_by=None):
    rules = []
    for r in results:
        for rule in r.ordered_statistics:
            base = ['None'] if len(rule.items_base) == 0 else list(rule.items_base)
            add = list(rule.items_add)
            rules.append([f'[{", ".join(base)}]', f'[{", ".join(add)}]',
                          round(rule.confidence, 2), round(rule.lift, 2)])

    table = prettytable.PrettyTable(field_names=['Rule', 'Confidence', 'Lift'])

    for r in rules:
        table.add_row([f'{r[0]} -> {r[1]}', r[2], r[3]])

    table.align = 'l'
    if sort_by is None:
        print(table)
    elif sort_by == 'confidence':
        print(table.get_string(sortby='Confidence', reversesort=True))
    elif sort_by == 'lift':
        print(table.get_string(sortby='Lift', reversesort=True))
    else:
        print(table.get_string(sortby='Rule'))


def single_run(filename, sep, min_support=0.1, min_confidence=0.0, min_lift=0.0, max_length=None):
    transactions = load_dataset(filename, sep)
    results = list(apriori(transactions, min_support=min_support, min_confidence=min_confidence, min_lift=min_lift,
                           max_length=max_length))
    return results


def analyse_time_on_conf(filename, sep):
    transactions = load_dataset(filename, sep)
    confidence = {}
    print('Analysing: dependence of time on confidence')
    for _ in tqdm(range(25)):
        for conf in np.linspace(0, 1, 41):
            if conf == 0:
                continue
            conf = round(conf, 3)
            if conf not in confidence:
                confidence[conf] = []

            t = time.time()
            list(apriori(transactions, min_support=0.01, min_confidence=conf, max_length=3))
            t = time.time() - t
            confidence[conf].append(t)
    resx, resy = [], []
    for conf, times in confidence.items():
        resx.append(conf)
        resy.append(sum(times)/len(times))

    plt.plot(resx, resy)
    plt.title('Dependence of time on confidence')
    plt.xlabel('Confidence')
    plt.ylabel('Time (s)')
    plt.grid()
    plt.savefig('time_on_conf.png')


def analyse_rules_amount_on_conf(filename, sep):
    transactions = load_dataset(filename, sep)
    confidence = {}

    print('Analysing: dependence of rules amount on confidence')
    for conf in tqdm(np.linspace(0, 1, 41)):
        conf = round(conf, 3)

        results = list(apriori(transactions, min_support=0.01, min_confidence=conf, max_length=3))

        amount = 0
        for r in results:
            amount += len(r.ordered_statistics)
        confidence[conf] = amount

    resx = []
    resy = []
    for conf, amount in confidence.items():
        resx.append(conf)
        resy.append(amount)

    plt.plot(resx, resy)
    plt.title('Dependence of rules amount on confidence')
    plt.xlabel('Confidence')
    plt.ylabel('Rules amount')
    plt.grid()
    plt.savefig('rules_amount_per_conf.png')


def main():
    # analyse_time_on_conf('retail.dat', ' ')
    # analyse_rules_amount_on_conf('retail.dat', ' ')
    # return

    results = single_run(
        filename='retail.dat',
        sep=' ',
        # The minimum support of relations (float). Default = 0.1
        min_support=0.03,
        # The minimum confidence of relations (float). Default = 0.0
        min_confidence=0.75,
        # The minimum lift of relations (float). Default = 0.0
        min_lift=0.0,
        # The maximum length of the relation (integer). Default = None
        max_length=None
    )
    # print_apriori_results(results, sort_by='support')  # sort_by: None, support, items
    print_apriori_results_rules(results, sort_by='confidence')  # sort_by: None, confidence, lift, rules


if __name__ == '__main__':
    main()
