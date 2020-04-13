import copy

import numpy as np
import matplotlib.pyplot as plt

from data_reader import DataProcessor
from models import Models
from workspace import People, PersonSet

data_path = 'corona_tested_individuals_ver_001.xlsx'
dp = DataProcessor(data_path)
dp.clean_data()
X_train, X_test, y_train, y_test = dp.split_data()
models = Models(X_train, y_train)
xgb_model = models.get_xgb_model()

data_by_day = dp.get_daily_data(7)

set_size = 5
graph_data_per_day = {}
for day, day_data in data_by_day.items():
    all_people = People(xgb_model, load=day_data)
    print(f"*********************************************************************")
    print(f"*********************************************************************")
    print(f"*********************************************************************")
    print(f"*********************************************************************")
    print(f"*********************************************************************")
    print(f"*********************************************************************")
    print(f"*********************************************************************")
    print(f"Day: {day}")
    print(f"Loaded: {len(all_people)} people")
    people_left = len(all_people)
    n_persons = set_size ** 2

    count_better = 0
    count_same = 0
    count_worse = 0
    errors = 0
    n_sick = 0
    all_tests = []
    all_tests_orig = []
    infection_buildings = np.zeros((set_size, set_size))
    infection_buildings_orig = np.zeros((set_size, set_size))
    all_infections = {}
    j = 0
    while people_left >= n_persons:
        # if people_left % 100 == 0:
        #     print(f"People left: {people_left}")
        Set = PersonSet(size=set_size)
        people_sample = all_people.get_people_sample(n_persons, no_reuse=True)
        for p in people_sample:
            # print(p)
            Set.add_person(p)
            people_left -= 1
            print(f"People left: {people_left}")

        OriginalSet = copy.deepcopy(Set)
        Set.arrange()
        # print(Set)
        Set.find()
        all_infections[j] = Set.view_infections()
        j += 1
        infection_buildings += Set.infection_set
        # print(f"Sicks found: {Set.n_found}")
        # print(f"Tests used: {Set.tests_used}")
        # print('\n')
        OriginalSet.build()
        # print(OriginalSet)
        OriginalSet.find()
        infection_buildings_orig += OriginalSet.infection_set
        # print(f"Sicks found: {OriginalSet.n_found}")
        # print(f"Tests used: {OriginalSet.tests_used}")

        if Set.ids_found == OriginalSet.ids_found:
            n_sick += Set.n_found
            all_tests.append(Set.tests_used)
            all_tests_orig.append(OriginalSet.tests_used)
            count_better += 1 if Set.tests_used < OriginalSet.tests_used else 0
            count_same += 1 if Set.tests_used == OriginalSet.tests_used else 0
            count_worse += 1 if Set.tests_used > OriginalSet.tests_used else 0
            # print(f"{Set.tests_used} Set vs {OriginalSet.tests_used} OriginalSet ({Set.n_found} sick)")
        else:
            errors += 1

    graph_data_per_day[day] = {'n_sick': n_sick,
                               'total_people': len(all_people),
                               'n_tests_prim': np.sum(all_tests_orig),
                               'n_tests': np.sum(all_tests),
                               }


def graph_days_barplot(graph_data_per_day):
    labels = [str(k)[:10] for k in list(graph_data_per_day.keys())]
    prims = [d['n_tests_prim'] for d in graph_data_per_day.values()]
    evens = [d['n_tests'] for d in graph_data_per_day.values()]
    nopool = [d['total_people'] for d in graph_data_per_day.values()]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    offset = 0.1

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width+offset, nopool, width-offset, label='No pooling')
    rects2 = ax.bar(x, prims, width-offset, label='Original pooling')
    rects3 = ax.bar(x + width-offset, evens, width-offset, label='Rearranged pooling')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Number of tests')
    ax.set_title('Number of tests by day and pooling method')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.legend()
    plt.show()

graph_days_barplot(graph_data_per_day)
