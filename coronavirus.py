import numpy as np
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from models import test_subjects, xgb_model, data_by_day


class Person:
    def __init__(self, id, is_infected, cough, fever, sore_throat, shortness_of_breath,
                 head_ache, age_60_and_above, gender, was_abroad, had_contact):
        self.id = id
        self.features = np.array([cough, fever, sore_throat, shortness_of_breath,
                 head_ache, age_60_and_above, gender, was_abroad, had_contact])
        self.feature_names = np.array(['cough', 'fever', 'sore_throat', 'shortness_of_breath',
                 'head_ache', 'age_60_and_above', 'gender', 'was_abroad', 'had_contact'])
        self.danger_level = self.set_danger_level()
        self.is_infected = is_infected
        # self.is_infected = self.set_infection()

    def __repr__(self):
        id_repr = f"""
                id = {self.id}
                """
        features_repr = ''
        for i in range(len(self.features)):
            features_repr += f'{self.feature_names[i]} = {self.features[i]}\n'
        more_repr = f"""
                danger_level = {self.danger_level}
                is_infected  = {self.is_infected}
                """
        return '\n'.join([id_repr, features_repr, more_repr])

    def set_danger_level(self):
        person_data = self.features.reshape((1, len(self.features)))
        return xgb_model.predict_proba(person_data)[0, 1] * 100
    #
    # def set_infection(self):
    #     return np.random.randint(1, 100) < self.danger_level


class People:
    def __init__(self, load=None):
        if load.any():
            self.people = self.load_people_from_csv(load)
        else:
            self.people = self.generate_random_people()
        self.used_people = []

    def __len__(self):
        return len(self.people)

    def load_people_from_csv(self, loaded_features):
        people = []
        for i in range(len(loaded_features)):
            p = loaded_features[i, :]
            people.append(Person(i, *p)) # * splits the list to individual parameters
        return people

    def generate_random_people(self):
        pass

    def get_people_sample(self, n, no_reuse=False):
        if no_reuse:
            choose_from = [p for p in self.people if p.id not in self.used_people]
            chosen = np.random.choice(choose_from, n, replace=False)
            for p in chosen:
                self.used_people.append(p.id)
            return chosen
        else:
            return np.random.choice(self.people, n, replace=False)


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
        # # Probably not possible because of low signal חכמולוגים של ויצמן
        # if sum(row_pool) + sum(col_pool) == 0:
        #     tests_used = 1
        #     return lst_of_sick, 0, tests_used
        # else:
        #     tests_used += 1

        row_pool = list(row_pool)
        line_rows = [i for i in range((len(row_pool))) if row_pool[i] == 1]

        col_pool = list(col_pool)
        line_col = [i for i in range(len(col_pool)) if col_pool[i] == 1]

        for row in line_rows:
            for col in line_col:
                if self.infection_set[row, col]:
                    lst_of_sick.append((row, col))
                if sum(row_pool) == 1 or sum(col_pool) == 1:
                    # אין צורך בבדיקות נוספות
                    continue
                tests_used += 1

        self.found_coords = lst_of_sick
        self.find_ids()
        self.n_found = len(lst_of_sick)
        self.tests_used = tests_used
        return lst_of_sick, self.n_found, tests_used

    def find_ids(self):
        for i, j in self.found_coords:
            self.ids_found.add(self.id_set[i,j])

# משחק בול פגיעה מהצבא
#שלב הבא סימולציה - ניקח לולאה ונעביר את הדבר הזה 10 פעמים, נסכום את כמות הפעולות בכל פעם ואת כמות החולים ונדע להגיד כמה חולים זיהינו ב360 פונטציאלים

if __name__ == "__main__":
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
    # all_people = People(load=test_subjects)
    graph_data_per_day = {}
    for day, day_data in data_by_day.items():
        all_people = People(load=day_data)

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

        graph_data_per_day[day] = {'n_sick': n_sick,
                                   'n_tests_prim': np.sum(all_tests_orig),
                                   'n_tests': np.sum(all_tests),
                                   }
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
        print(f"Sum tests before {np.sum(all_tests_orig)}\nMean tests before {np.mean(all_tests_orig)} \nSD tests before {np.std(all_tests_orig)}")
        print(f"Sum tests after {np.sum(all_tests)}\nMean tests after {np.mean(all_tests)} \nSD tests after {np.std(all_tests)}")
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