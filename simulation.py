import copy

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from coronavirus import People, PersonSet
from models import test_subjects

count_better = 0
count_same = 0
count_worse = 0
errors = 0
n_sick = 0
all_tests = []
all_tests_orig = []
n_iterations = 1
set_size = 5
infection_buildings = np.zeros((set_size, set_size))
infection_buildings_orig = np.zeros((set_size, set_size))
all_infections = {}
# all_people = People(load=all_subjects)
all_people = People(load=test_subjects)
graph_data_per_day = {}


n_persons = set_size ** 2
for j in tqdm(range(n_iterations)):
    Set = PersonSet(size=set_size)
    people_sample = all_people.get_people_sample(n_persons, no_reuse=True)
    for p in people_sample:
        # print(p)
        Set.add_person(p)

    OriginalSet = copy.deepcopy(Set)
    Set.arrange()
    # print(Set)
    Set.find()
    all_infections[j] = Set.view_infections()
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

good_iterations = n_iterations - errors
print("\nSummary\n----------")
print(
    f"Ran {n_iterations} iterations on samples of {n_persons} people ({set_size}X{set_size})")
print(f"Errors: {errors}")
print(f"Average of {n_sick / good_iterations} infected per sample\n")

perc_better = count_better * 100.0 / good_iterations
perc_same = count_same * 100.0 / good_iterations

print(
    f"Set better than or same as OriginalSet in {(count_better + count_same) * 100.0 / good_iterations} % of times! (better: {perc_better}, same: {perc_same})")

print(
    f"Set worse than OriginalSet in {count_worse * 100.0 / good_iterations} % of times!")

print(f"All sicks: {n_sick}")
print(
    f"Sum tests before {np.sum(all_tests_orig)}\nMean tests before {np.mean(all_tests_orig)} \nSD tests before {np.std(all_tests_orig)}")
print(
    f"Sum tests after {np.sum(all_tests)}\nMean tests after {np.mean(all_tests)} \nSD tests after {np.std(all_tests)}")
print(Set)

for k, v in all_infections.items():
    print(f"iteration {k + 1}")
    print(v)

print(infection_buildings)

fig = plt.figure(figsize=(16, 8))
buildings = {'Re-ordered': infection_buildings,
         'Original order': infection_buildings_orig}
fig_n = 0
for name, building in buildings.items():
    fig_n += 1
    ax1 = fig.add_subplot(2, 2, fig_n, projection='3d')

    _x = np.arange(building.shape[0])
    _y = np.arange(building.shape[1])
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()

    dz = building.ravel()
    z = np.zeros_like(dz)
    dx = dy = 1
    ax1.bar3d(x, y, z, dx, dy, dz, shade=True, alpha=0.2)
    ax1.set_title(f'CoronaCity - {name}')

    fig_n += 1
    ax2 = fig.add_subplot(2, 2, fig_n)
    ax2.set_title(f'Heatmap - {name}')
    # cmap='hot'
    im = ax2.imshow(building, cmap='viridis', interpolation='nearest',
                    vmax=np.max(infection_buildings))
    plt.colorbar(im)

plt.show()
