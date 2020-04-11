import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import statistics
from tqdm import tqdm


class Person:
    def __init__(self, id, age, has_background_disease, days_since_hul,
                 is_quarantined, area_danger_level, blood_type, bias):
        self.id = id
        self.age = age
        self.has_background_disease = has_background_disease
        self.days_since_hul = days_since_hul
        self.is_quarantined = is_quarantined
        self.area_danger_level = area_danger_level
        self.blood_type = blood_type
        self.bias = bias
        self.danger_level = self.set_danger_level()
        self.is_infected = self.set_infection()

    def __repr__(self):
        return f"""
                id = {self.id}
                age = {self.age}
                has_background_disease = {self.has_background_disease}
                days_since_hul = {self.days_since_hul}
                is_quarantined = {self.is_quarantined}
                area_danger_level = {self.area_danger_level}
                blood_type = {self.blood_type}
                danger_level = {self.danger_level}
                is_infected  = {self.is_infected}
                """

    def set_danger_level(self):
        age = self.age
        has_background_disease = 75 if self.has_background_disease else 25
        days_since_hul = 80 if self.days_since_hul < 14 else 20
        is_quarantined = 75 if self.has_background_disease else 25
        area_danger_level = self.area_danger_level
        blood_type = 55 if self.blood_type == 'A' else 45 if self.blood_type == 'O' else 50
        danger_level = np.mean([age, has_background_disease, days_since_hul,
                                is_quarantined, area_danger_level, blood_type, self.bias]
                               )
        return np.max([danger_level, np.random.randint(DL_LOW, DL_HIGH)])

    def set_infection(self):
        return np.random.randint(1, 100) < self.danger_level


class PersonSet:
    def __init__(self, size):
        self.size = size
        self.persons = []
        self.id_set = np.zeros([size, size])
        self.infection_set = np.zeros([size, size])
        self.danger_set = np.zeros([size, size])
        self.found_coords = []
        self.n_found = 0
        self.tests_used = 0
        self.ids_found = set([])

    def __repr__(self):
        id_view = self.view_ids()
        inf_view = self.view_infections()
        dng_view = self.view_danger_levels()
        return "ids\n" + id_view + "\n\ninfections\n" + inf_view + "\n\ndanger level\n" + dng_view

    def view_ids(self):
        id_view = ""
        n_fill = len(str(self.size ** 2))
        for i in range(self.size):
            for j in range(self.size):
                id_view = id_view + f" {str(int(self.id_set[i, j])).rjust(n_fill, '0')} "
            id_view = id_view + "\n"
        return id_view

    def view_infections(self):
        inf_view = ""
        for i in range(self.size):
            for j in range(self.size):
                inf_view = inf_view + f" {'*' if self.infection_set[i, j] else 'O'} "
            inf_view = inf_view + "\n"
        return inf_view

    def view_danger_levels(self):
        dng_view = ""
        for i in range(self.size):
            for j in range(self.size):
                dng_view = dng_view + f" {np.round(self.danger_set[i, j])} "
            dng_view = dng_view + "\n"
        return dng_view

    def add_person(self, person):
        if person.id not in [person.id for person in self.persons]:
            self.persons.append(person)
        else:
            print(f"person id {person.id} already in set")

    def sort_persons_by_danger(self):
        self.persons = sorted(self.persons, key=lambda p: p.danger_level, reverse=True)

    def build(self):
        for i in range(self.size):
            for j in range(self.size):
                person = self.persons[j * self.size + i]
                self.id_set[i, j] = person.id
                self.infection_set[i, j] = person.is_infected
                self.danger_set[i, j] = person.danger_level

    def arrange(self):
        self.sort_persons_by_danger()
        self.build()

    def pool(self):
        # add row and columns
        row_sums = np.zeros((self.size, 1))
        col_sums = np.zeros((1, self.size))
        compute_rows = np.append(self.infection_set, row_sums, axis=1)
        compute_cols = np.append(self.infection_set, col_sums, axis=0)

        # found rows and columns of positive test
        for i in range(self.size):
            if sum(compute_rows[i, :]) > 0:
                compute_rows[i, [self.size]] = 1
        row_pool = compute_rows[:, self.size]

        for j in range(self.size):
            if sum(compute_cols[:, j]) > 0:
                compute_cols[self.size, [j]] = 1
        col_pool = compute_cols[self.size, :]
        return row_pool, col_pool

    def find(self):
        tests_used = 2 * self.size
        lst_of_sick = []
        row_pool, col_pool = self.pool()

        row_pool = list(row_pool)
        line_rows = [i for i in range((len(row_pool))) if row_pool[i] == 1]

        col_pool = list(col_pool)
        line_col = [i for i in range(len(col_pool)) if col_pool[i] == 1]

        # חולה יחיד
        if sum(row_pool) == 1 and sum(col_pool) == 1:
            a = (row_pool.index(max(row_pool)), col_pool.index(max(col_pool)))
            lst_of_sick.append(a)

        # תבנית קלה של משלים במידה והבחירה הראשונה לא הצליחה אם היא כן הצליחה חייבים לבדוק את שניהם
        # לחשוב לסנן את הרשימנות בדיקה לאלו שיש בהם 1- ולקחת רק את האינדקס שהוא מייצג
        else:
            for row in line_rows:
                for col in line_col:
                    tests_used += 1
                    if self.infection_set[row, col]:
                        lst_of_sick.append((row, col))
        self.found_coords = lst_of_sick
        self.find_ids()
        self.n_found = len(lst_of_sick)
        self.tests_used = tests_used
        return lst_of_sick, self.n_found, tests_used

    def find_ids(self):
        for i, j in self.found_coords:
            self.ids_found.add(self.id_set[i,j])


# k =1
#
# big_count = 0
# big_count2 = 0
# big_sick = 0
# list_of_count = []
#
#
# for i in range(1000000):
#     lst_of_sick, count_of_test, matrix_of_sick = create_sets()
#     big_count += count_of_test
#     big_count2 += count_of_test
#     list_of_count.append(big_count2)
#     big_count2 = 0
#     big_sick += len(lst_of_sick)
#
# e = statistics.mean(list_of_count)
# sd = statistics.stdev(list_of_count)
# print("the mean of simulatin is: ",e)
# print("the sd of simulatin is: ",sd)
# k=56


# משחק בול פגיעה מהצבא

#שלב הבא סימולציה - ניקח לולאה ונעביר את הדבר הזה 10 פעמים, נסכום את כמות הפעולות בכל פעם ואת כמות החולים ונדע להגיד כמה חולים זיהינו ב360 פונטציאלים

if __name__ == "__main__":
    count_better = 0
    count_same = 0
    count_worse = 0
    errors = 0
    n_sick = 0
    all_tests = []
    n_iterations = 100
    DL_LOW = 2
    DL_HIGH = 8
    BIAS = -230
    set_size = 6
    infection_buildings = np.zeros((set_size, set_size))
    infection_buildings_orig = np.zeros((set_size, set_size))
    all_infections = {}
    for j in tqdm(range(n_iterations)):
        Set = PersonSet(size=set_size)
        n_persons = set_size ** 2
        for i in range(1, n_persons + 1):
            age = np.random.randint(1, 100)
            has_background_disease = np.random.choice([True, False], p=[0.25, 0.75])
            days_since_hul = np.random.randint(1, 100)
            is_quarantined = np.random.choice([True, False], p=[0.1, 0.9])
            area_danger_level = np.random.randint(1, 100)
            blood_type = np.random.choice(['A', 'B', 'O', 'AB'])
            bias = BIAS
            p = Person(i, age, has_background_disease, days_since_hul,
                       is_quarantined, area_danger_level, blood_type, bias)
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
            count_better += 1 if Set.tests_used < OriginalSet.tests_used else 0
            count_same += 1 if Set.tests_used == OriginalSet.tests_used else 0
            count_worse += 1 if Set.tests_used > OriginalSet.tests_used else 0
            # print(f"{Set.tests_used} Set vs {OriginalSet.tests_used} OriginalSet ({Set.n_found} sick)")
        else:
            errors += 1

    good_iterations = n_iterations - errors
    print("\nSummary\n----------")
    print(f"Ran {n_iterations} iterations on samples of {n_persons} people ({set_size}X{set_size})")
    print(f"Errors: {errors}")
    print(f"Average of {n_sick / good_iterations} infected per sample\n")

    perc_better = count_better * 100.0 / good_iterations
    perc_same = count_same * 100.0 / good_iterations

    print(f"Set better than or same as OriginalSet in {(count_better + count_same) * 100.0/ good_iterations} % of times! (better: {perc_better}, same: {perc_same})")

    print(f"Set worse than OriginalSet in {count_worse * 100.0/ good_iterations} % of times!")

    print(f"All sicks: {n_sick}")
    print(f"Sum tests {np.sum(all_tests)}\nMean tests {np.mean(all_tests)} \nSD tests {np.std(all_tests)}")
    print(Set)

    for k, v in all_infections.items():
        print(f"iteration {k+1}")
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
    im = ax2.imshow(building, cmap='viridis', interpolation='nearest', vmax=np.max(infection_buildings))
    plt.colorbar(im)

plt.show()

