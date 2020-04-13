import numpy as np


class Person:
    def __init__(self, model, id, is_infected, cough, fever, sore_throat, shortness_of_breath,
                 head_ache, age_60_and_above, gender, was_abroad, had_contact):
        self.id = id
        self.features = np.array([cough, fever, sore_throat, shortness_of_breath,
                 head_ache, age_60_and_above, gender, was_abroad, had_contact])
        self.feature_names = np.array(['cough', 'fever', 'sore_throat', 'shortness_of_breath',
                 'head_ache', 'age_60_and_above', 'gender', 'was_abroad', 'had_contact'])
        self.danger_level = self.set_danger_level(model)
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

    def set_danger_level(self, model):
        person_data = self.features.reshape((1, len(self.features)))
        return model.predict_proba(person_data)[0, 1] * 100
    #
    # def set_infection(self):
    #     return np.random.randint(1, 100) < self.danger_level


class People:
    def __init__(self, model=None, load=None):
        self.model = model
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
            people.append(Person(self.model, i, *p)) # * splits the list to individual parameters
        return people

    def generate_random_people(self):
        return []

    def get_people_sample(self, n, no_reuse=False):
        if no_reuse:
            choose_from = [p for p in self.people if p.id not in self.used_people]
            chosen = np.random.choice(choose_from, n, replace=False)
            for p in chosen:
                self.used_people.append(p.id)
            return chosen
        else:
            return np.random.choice(self.people, n, replace=False)

    def sort_by_risk(self, reverse=True):
        return sorted(self.people, key=lambda x: x.danger_level, reverse=reverse)

    def get_people_list(self, randomize=False):
        if randomize:
            return np.random.permutation(self.people)
        return self.people


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

    def derrange(self):
        np.random.shuffle(self.persons)

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
