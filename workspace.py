import numpy as np
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

from data_reader import DataProcessor
from models import Models, DEFAULT_MODEL
from person_classes import People, PersonSet


class Workspace:
    def __init__(self, data_path, set_size, model=DEFAULT_MODEL, flip=False):
        self.dp = DataProcessor(data_path)
        self.dp.clean_data(flip)
        self.n_days_available = len(self.dp.all_dates)
        X_train, X_test, y_train, y_test = self.dp.split_data()
        models = Models(X_train, y_train)
        self.model_name = model
        self.model = models.get_model(model)

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

    def work(self, people_sample, randomize_orig=False):
        Set = PersonSet(size=self.set_size)
        for p in people_sample:
            # print(p)
            Set.add_person(p)

        OriginalSet = copy.deepcopy(Set)
        if randomize_orig:
            OriginalSet.derrange()
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

    def summary(self, n, visualize_tables=False):
        good_iterations = n - self.errors
        print("\nSummary\n----------")
        print(
            f"Ran {n} iterations on samples of {self.n_persons} people ({self.set_size}X{self.set_size})")
        total_checked = n * self.n_persons
        print(f"Total people checked: {total_checked}")
        print(f"Errors: {self.errors}")
        avg_sick = self.n_sick / good_iterations
        print(f"Average of {avg_sick} infected per sample\n")

        perc_better = self.count_better * 100.0 / good_iterations
        perc_same = self.count_same * 100.0 / good_iterations

        print(
            f"Set better than or same as OriginalSet in {(self.count_better + self.count_same) * 100.0 / good_iterations} % of times! (better: {perc_better}, same: {perc_same})")

        print(
            f"Set worse than OriginalSet in {self.count_worse * 100.0 / good_iterations} % of times!")

        print(f"All sicks: {self.n_sick}")
        sum_tests_before = np.sum(self.all_tests_orig)
        mean_tests_before = np.mean(self.all_tests_orig)
        std_tests_before = np.std(self.all_tests_orig)
        print(
            f"Sum tests before {sum_tests_before}\nMean tests before {mean_tests_before} \nSD tests before {std_tests_before}\n")
        sum_tests_after = np.sum(self.all_tests)
        mean_tests_after = np.mean(self.all_tests)
        std_tests_after = np.std(self.all_tests)
        print(
            f"Sum tests after {sum_tests_after}\nMean tests after {mean_tests_after} \nSD tests after {std_tests_after}")
        if visualize_tables:
            for i, v in enumerate(self.all_infections):
                print(f"iteration {i + 1}")
                print(v)

        print(self.infection_buildings)
        return {'n': n,
                'set_size': self.set_size,
                'total_checked': total_checked,
                'errors': self.errors,
                'avg_sick': avg_sick,
                'perc_better': perc_better,
                'perc_same': perc_same,
                'sum_tests_before': sum_tests_before,
                'mean_tests_before': mean_tests_before,
                'std_tests_before': std_tests_before,
                'sum_tests_after': sum_tests_after,
                'mean_tests_after': mean_tests_after,
                'std_tests_after': std_tests_after,
                'all_infections': self.all_infections
                }

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
            # cmap='Blues'
            # cmap='viridis'
            im = ax2.imshow(building, cmap='viridis', interpolation='nearest',
                            vmax=np.max(self.infection_buildings))
            plt.colorbar(im)

        plt.show()

    def graph_days_barplot(self, graph_data_per_day, matrices_sorted=True, verbose=False):
        if verbose:
            labels = [str(k)[:10] for k in list(graph_data_per_day.keys())]
            prims_sorted = [d['n_tests_prim_sorted'] for d in graph_data_per_day.values()]
            evens_sorted = [d['n_tests_sorted'] for d in graph_data_per_day.values()]
            prims_unsorted = [d['n_tests_prim_unsorted'] for d in graph_data_per_day.values()]
            evens_unsorted = [d['n_tests_unsorted'] for d in graph_data_per_day.values()]
            nopool = [d['total_people'] for d in graph_data_per_day.values()]

            width = 0.35  # the width of the bars
            offset = 0.1
            x = 2*np.arange(len(labels))  # the label locations
            fig, ax = plt.subplots()
            rects1 = ax.bar(x + offset, nopool, width - offset,
                            label='No pooling')
            rects2 = ax.bar(x + width + offset, prims_unsorted, width - offset,
                            label='Original pooling - unsorted')
            rects2_ = ax.bar(x + 2*width + offset, prims_sorted, width - offset,
                             label='Original pooling - sorted')
            rects3 = ax.bar(x + 3*width + offset, evens_unsorted, width - offset,
                            label='Rearranged pooling - unsorted')
            rects3_ = ax.bar(x + 4*width + offset, evens_sorted, width - offset,
                            label='Rearranged pooling - sorted')

            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_ylabel('Number of tests')
            ax.set_title(f'Number of tests by day and pooling method - model: {self.model_name}')
            ax.set_xticks(x + 2*width + offset)
            ax.set_xticklabels(labels, rotation=45)
            ax.legend()
            plt.show()
            return prims_sorted, evens_sorted, prims_unsorted, evens_unsorted, \
                   nopool
        else:
            suffix = '_sorted' if matrices_sorted else '_unsorted'
            labels = [str(k)[:10] for k in list(graph_data_per_day.keys())]
            prims = [d['n_tests_prim' + suffix] for d in graph_data_per_day.values()]
            evens = [d['n_tests' + suffix] for d in graph_data_per_day.values()]
            # evens = [d['n_tests_unsorted'] for d in graph_data_per_day.values()]
            nopool = [d['total_people'] for d in graph_data_per_day.values()]

            width = 0.35  # the width of the bars
            offset = 0.1
            x = np.arange(len(labels))  # the label locations
            fig, ax = plt.subplots()
            rects1 = ax.bar(x - width + offset, nopool, width - offset,
                            label='No pooling')
            rects2 = ax.bar(x, prims, width - offset, label='Original pooling')
            rects3 = ax.bar(x + width - offset, evens, width - offset,
                            label='Rearranged pooling')

            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_ylabel('Number of tests')
            ax.set_title(f'Number of tests by day and pooling method - model: {self.model_name}')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45)
            ax.legend()
            plt.show()
            return nopool, prims, evens

    def daily(self, date_input, end_date=np.datetime64('today'),
              matrices_sorted=True, display_other=False):
        # TODO: specific days instead of days_back
        data_by_day = self.dp.get_daily_data(date_input, end_date)
        graph_data_per_day = {}
        for day, day_data in data_by_day.items():
            all_people = People(model=self.model, load=day_data)
            print(6 * f"*********************************************************************\n")
            print(f"Day: {day}")
            print(f"Loaded: {len(all_people)} people")
            graph_data_per_day[day] = self._one_day_work(all_people,
                                                         matrices_sorted)
            if display_other:
                graph_data_per_day[day].update(self._one_day_work(all_people,
                                                                  not matrices_sorted))

        bar_data = self.graph_days_barplot(graph_data_per_day,
                                           matrices_sorted=matrices_sorted,
                                           verbose=display_other)
        if display_other:
            prims_sorted, evens_sorted, prims_unsorted, evens_unsorted, \
            nopool = bar_data
            all_days = list(graph_data_per_day.keys())
            print(f"Days: {all_days}")
            print(f"No pooling sorted: {nopool}")
            print(f"Original pooling sorted: {prims_sorted}")
            print(f"Rearranged pooling sorted: {evens_sorted}")
            print(f"Original pooling unsorted: {prims_unsorted}")
            print(f"Rearranged pooling unsorted: {evens_unsorted}")

            ratio = np.array(evens_sorted) / np.array(prims_unsorted)
            goal = 0.6
            great_days = np.where(ratio < goal)[0]
            if great_days:
                print(great_days)
                print(f'Number of days with improvement more than {goal * 100}%: {len(great_days)}')
                for day in great_days:
                    print(all_days[day])

        else:
            nopool, prims, evens = bar_data
            print(f"Days: {list(graph_data_per_day.keys())}")
            print(f"No pooling: {nopool}")
            print(f"Original pooling: {prims}")
            print(f"Rearranged pooling: {evens}")

    def _one_day_work(self, all_people, matrices_sorted):
        self.reset_variables()
        matrices, unused_people = self.arrange_person_matrices(
            all_people, risk_sorted=not matrices_sorted)
        for matrix in tqdm(matrices):
            self.work(matrix, randomize_orig=True)
        suffix = '_sorted' if matrices_sorted else '_unsorted'
        return {'n_sick': self.n_sick,
                'total_people': len(all_people),
                'n_tests_prim' + suffix: np.sum(self.all_tests_orig),
                'n_tests' + suffix: np.sum(self.all_tests),
                }

    def sample_test_set(self, n_iterations):
        self.reset_variables()
        # subjects = self.dp.get_all_data()
        subjects = self.dp.get_test_data()
        all_people = People(model=self.model, load=subjects)
        for j in tqdm(range(n_iterations)):
            people_sample = all_people.get_people_sample(self.n_persons, no_reuse=True)
            self.work(people_sample)
        self.summary(n_iterations)
        self.graph_heatmap()

    def examine_entire_test_set(self):
        self.reset_variables()
        # subjects = self.dp.get_all_data()
        subjects = self.dp.get_test_data()
        all_people = People(model=self.model, load=subjects)
        matrices, unused_people = self.arrange_person_matrices(all_people)
        for matrix in tqdm(matrices):
            self.work(matrix, randomize_orig=True)
        print(f"{len(unused_people)} were omitted")
        self.summary(len(matrices))
        self.graph_heatmap()

    def examine_simulation_set(self, n, p_sick):
        self.reset_variables()
        all_people = People()
        all_people.generate_random_people(n, p_sick)
        matrices, unused_people = self.arrange_person_matrices(all_people)
        for matrix in tqdm(matrices):
            self.work(matrix, randomize_orig=True)
        print(f"{len(unused_people)} were omitted")
        self.summary(len(matrices))
        self.graph_heatmap()

    def compare_simulations(self, n, p_sicks, curve=False, disp_nopool=True):
        summaries = {}
        for p_sick in p_sicks:
            self.reset_variables()
            all_people = People()
            all_people.generate_random_people(n, p_sick)
            matrices, unused_people = self.arrange_person_matrices(all_people)
            for matrix in tqdm(matrices):
                self.work(matrix, randomize_orig=True)
            print(f"{len(unused_people)} were omitted")
            summaries[p_sick] = self.summary(len(matrices))
        fig, ax = plt.subplots()
        nopool = [n for p_sick in p_sicks]
        prims_unsorted = [summaries[p_sick]['sum_tests_before'] for p_sick in p_sicks]
        evens_sorted = [summaries[p_sick]['sum_tests_after'] for p_sick in p_sicks]

        width = 0.30  # the width of the bars
        offset = 0.1
        x = np.arange(len(p_sicks))  # the label locations

        if curve:
            width = 0  # the width of the bars
            offset = 0
            # x = np.arange(len(p_sicks))
            if disp_nopool:
                ax.plot(x, nopool, 'r', label='No pooling')
            ax.plot(x, prims_unsorted, 'g', label='Original pooling')
            ax.plot(x, evens_sorted, 'b', label='Rearranged pooling')
            plt.grid()

        else:
            n_bars = 2
            if disp_nopool:
                # n_bars += 1
                rects1 = ax.bar(x + offset, nopool, width - offset,
                                label='No pooling')
            rects2 = ax.bar(x + width + offset, prims_unsorted, width - offset,
                            label='Original pooling')
            rects3_ = ax.bar(x + n_bars * width + offset, evens_sorted, width - offset,
                             label='Rearranged pooling')

        ax.set_ylabel('Number of tests')
        ax.set_title(
            f'Number of tests by infection rate and pooling method ({n} people)\n')
        ax.set_xticks(x + width + offset)
        ax.set_xticklabels(p_sicks, rotation=45)
        ax.set_xlabel('Infection rate')
        ax.legend()
        plt.show()


    def arrange_person_matrices(self, people, risk_sorted=True):
        if risk_sorted:
            people_order = people.sort_by_risk()
        else:
            people_order = people.get_people_list(randomize=False)
        people_ids = [p.id for p in people_order]
        matrix_size = self.set_size ** 2
        n_matrices = len(people) // matrix_size
        n_left = len(people) % matrix_size
        pepole_left = people_ids[-n_left:]

        person_matrices = []
        for i in range(n_matrices):
            arranged_people = [people_order[n_matrices * j + i]
                               for j in range(matrix_size)]
            person_matrices.append(arranged_people)

        print(f'len(person_matrices): {len(person_matrices)}')

        return person_matrices, pepole_left

# משחק בול פגיעה מהצבא
#שלב הבא סימולציה - ניקח לולאה ונעביר את הדבר הזה 10 פעמים, נסכום את כמות הפעולות בכל פעם ואת כמות החולים ונדע להגיד כמה חולים זיהינו ב360 פונטציאלים


if __name__ == "__main__":
    # data_path = 'data/corona_tested_individuals_ver_003.xlsx'
    data_path = 'data/corona_tested_individuals_ver_0036.csv'
    # model_names = ['xgb', 'logreg', 'bayes', 'forest']
    # for model_name in model_names:
    flip = False if int(data_path.split('.')[0][-3:]) > 5 else True
    ws = Workspace(data_path, set_size=6, model='xgb', flip=flip)
    # ws.sample_test_set(n_iterations=10)
    date_input = 120
    # date_input = '2020-03-11' #27
    # end_date = '2020-03-31'

    # date_input = '2020-04-01'
    # end_date = '2020-04-30'
    # date_input = ['2020-03-28', '2020-03-30', '2020-04-02']
    # ws.daily(date_input=date_input, end_date=end_date, matrices_sorted=True, display_other=False)
    # ws.daily(date_input=date_input, matrices_sorted=True, display_other=False)
    ws.examine_entire_test_set()
    # ws.examine_simulation_set(10000, 0.05)
    # ws.compare_simulations(10000, [0.01, 0.05, 0.10, 0.15, 0.2], curve=True, disp_nopool=True)
