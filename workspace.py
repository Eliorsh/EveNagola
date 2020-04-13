import numpy as np
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

from data_reader import DataProcessor
from models import Models
from person_classes import People, PersonSet


class Workspace:
    def __init__(self, data_path, set_size):
        self.dp = DataProcessor(data_path)
        self.dp.clean_data()
        self.n_days_available = len(self.dp.all_dates)
        X_train, X_test, y_train, y_test = self.dp.split_data()
        models = Models(X_train, y_train)
        self.model = models.get_xgb_model()

        self.set_size = set_size
        self.n_persons = set_size ** 2
        self.reset_variables()

    def reset_variables(self):
        self.count_better = 0
        self.count_same = 0
        self.count_worse = 0
        self.errors = 0
        self.n_sick = 0
        self.all_tests = []
        self.all_tests_orig = []
        self.infection_buildings = np.zeros((self.set_size, self.set_size))
        self.infection_buildings_orig = np.zeros((self.set_size, self.set_size))
        self.all_infections = []

    def work(self, all_people):
        Set = PersonSet(size=self.set_size)
        people_sample = all_people.get_people_sample(self.n_persons, no_reuse=True)
        for p in people_sample:
            # print(p)
            Set.add_person(p)

        OriginalSet = copy.deepcopy(Set)
        Set.arrange()
        # print(Set)
        Set.find()
        self.all_infections.append(Set.view_infections())
        self.infection_buildings += Set.infection_set
        # print(f"Sicks found: {Set.n_found}")
        # print(f"Tests used: {Set.tests_used}")
        # print('\n')
        OriginalSet.build()
        # print(OriginalSet)
        OriginalSet.find()
        self.infection_buildings_orig += OriginalSet.infection_set
        # print(f"Sicks found: {OriginalSet.n_found}")
        # print(f"Tests used: {OriginalSet.tests_used}")

        if Set.ids_found == OriginalSet.ids_found:
            self.n_sick += Set.n_found
            self.all_tests.append(Set.tests_used)
            self.all_tests_orig.append(OriginalSet.tests_used)
            self.count_better += 1 if Set.tests_used < OriginalSet.tests_used else 0
            self.count_same += 1 if Set.tests_used == OriginalSet.tests_used else 0
            self.count_worse += 1 if Set.tests_used > OriginalSet.tests_used else 0
            # print(f"{Set.tests_used} Set vs {OriginalSet.tests_used} OriginalSet ({Set.n_found} sick)")
        else:
            self.errors += 1

    def summary(self, n):
        good_iterations = n - self.errors
        print("\nSummary\n----------")
        print(
            f"Ran {n} iterations on samples of {self.n_persons} people ({self.set_size}X{self.set_size})")
        print(f"Errors: {self.errors}")
        print(f"Average of {self.n_sick / good_iterations} infected per sample\n")

        perc_better = self.count_better * 100.0 / good_iterations
        perc_same = self.count_same * 100.0 / good_iterations

        print(
            f"Set better than or same as OriginalSet in {(self.count_better + self.count_same) * 100.0 / good_iterations} % of times! (better: {perc_better}, same: {perc_same})")

        print(
            f"Set worse than OriginalSet in {self.count_worse * 100.0 / good_iterations} % of times!")

        print(f"All sicks: {self.n_sick}")
        print(
            f"Sum tests before {np.sum(self.all_tests_orig)}\nMean tests before {np.mean(self.all_tests_orig)} \nSD tests before {np.std(self.all_tests_orig)}")
        print(
            f"Sum tests after {np.sum(self.all_tests)}\nMean tests after {np.mean(self.all_tests)} \nSD tests after {np.std(self.all_tests)}")

        for i, v in enumerate(self.all_infections):
            print(f"iteration {i + 1}")
            print(v)

        print(self.infection_buildings)

    def graph_heatmap(self):
        fig = plt.figure(figsize=(16, 8))
        buildings = {'Re-ordered': self.infection_buildings,
                     'Original order': self.infection_buildings_orig}
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
                            vmax=np.max(self.infection_buildings))
            plt.colorbar(im)

        plt.show()

    @staticmethod
    def graph_days_barplot(graph_data_per_day):
        labels = [str(k)[:10] for k in list(graph_data_per_day.keys())]
        prims = [d['n_tests_prim'] for d in graph_data_per_day.values()]
        evens = [d['n_tests'] for d in graph_data_per_day.values()]
        nopool = [d['total_people'] for d in graph_data_per_day.values()]

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars
        offset = 0.1

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width + offset, nopool, width - offset,
                        label='No pooling')
        rects2 = ax.bar(x, prims, width - offset, label='Original pooling')
        rects3 = ax.bar(x + width - offset, evens, width - offset,
                        label='Rearranged pooling')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Number of tests')
        ax.set_title('Number of tests by day and pooling method')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.legend()
        plt.show()

    def daily(self, days_back):
        data_by_day = self.dp.get_daily_data(days_back)
        graph_data_per_day = {}
        for day, day_data in data_by_day.items():
            all_people = People(model=self.model, load=day_data)
            print(6 * f"*********************************************************************\n")
            print(f"Day: {day}")
            print(f"Loaded: {len(all_people)} people")
            people_left = len(all_people)
            self.reset_variables()
            while people_left >= self.n_persons:
                self.work(all_people)
                people_left -= self.n_persons
                print(f"people_left: {people_left}")
            graph_data_per_day[day] = {'n_sick': self.n_sick,
                                       'total_people': len(all_people),
                                       'n_tests_prim': np.sum(self.all_tests_orig),
                                       'n_tests': np.sum(self.all_tests),
                                       }
        self.graph_days_barplot(graph_data_per_day)

    def simulate(self, n_iterations):
        self.reset_variables()
        # subjects = self.dp.get_all_data()
        subjects = self.dp.get_test_data()
        all_people = People(model=self.model, load=subjects)
        for j in tqdm(range(n_iterations)):
            self.work(all_people)
        self.summary(n_iterations)
        self.graph_heatmap()

# משחק בול פגיעה מהצבא
#שלב הבא סימולציה - ניקח לולאה ונעביר את הדבר הזה 10 פעמים, נסכום את כמות הפעולות בכל פעם ואת כמות החולים ונדע להגיד כמה חולים זיהינו ב360 פונטציאלים


if __name__ == "__main__":
    data_path = 'corona_tested_individuals_ver_001.xlsx'
    ws = Workspace(data_path, set_size=5)
    # ws.simulate(n_iterations=100)
    ws.daily(days_back=1)
