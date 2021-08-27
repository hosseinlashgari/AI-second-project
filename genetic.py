import random
import copy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style


class Game:
    def __init__(self, level: str):
        self.level = level
        self.level_len = len(level)
        self.G_number = 0
        self.M_number = 0
        for step in level:
            if step == 'G':
                self.G_number += 1
            elif step == 'M':
                self.M_number += 1

    def get_score(self, string: str, mode: int = 0):
        score = 0
        max_dis = 0
        current_dis = 0
        flag = True

        for step in range(0, self.level_len+1):
            if step == self.level_len:
                max_dis = max(max_dis, current_dis)
                if string[step-1] == '1' and string[step-2] != '1':
                    score += 1
                continue
            current_step = self.level[step]
            if step != 0:
                prev_step = self.level[step-1]
            else:
                prev_step = None
            if step != self.level_len-1:
                next_step = self.level[step+1]
            else:
                next_step = None

            if current_step == '_':
                current_dis += 1
                if prev_step:
                    if string[step-1] != '0' and not (next_step == 'G' and string[step-1] == '1'):
                        score -= 1
                continue
            if current_step == 'G':
                if step > 1 and string[step-2] == '1' and prev_step != 'L' and string[step-1] == '0':
                    current_dis += 1
                    score += 2
                elif step > 1 and string[step-1] == '1' and string[step-2] != '1':
                    current_dis += 1
                elif step <= 1 and string[step-1] == '1':
                    current_dis += 1
                else:
                    max_dis = max(max_dis, current_dis)
                    current_dis = 0
                    flag = False
                continue
            if current_step == 'L':
                if step > 1 and string[step-1] == '2' and string[step-2] != '1':
                    current_dis += 1
                elif step <= 1 and string[step-1] == '2':
                    current_dis += 1
                else:
                    max_dis = max(current_dis, max_dis)
                    current_dis = 0
                    flag = False
                continue
            if current_step == 'M':
                current_dis += 1
                if string[step-1] == '0':
                    score += 2
                continue
        if mode == 0:
            if flag:
                score = ((score + 10) / (2 * (self.M_number + self.G_number) + 11) + 1) * max_dis
            else:
                score = (score / (2 * (self.M_number + self.G_number) + 11) + 1) * max_dis
        elif mode == 1:
            score = (score / (2 * (self.M_number + self.G_number) + 1) + 1) * max_dis
        return flag, score


class Chromosome:
    def __init__(self, actions: str = "", flag: bool = False, score: float = -1000):
        self.string = actions
        self.len = len(actions)
        self.flag = flag
        self.score = score


class Generation:
    def __init__(self, pop: list):
        self.members = pop
        self.number = len(pop)
        self.best = pop[0]
        self.worst = pop[0]
        self.flag = False
        avr = 0
        for ch in pop:
            avr += ch.score
            if ch.score < self.worst.score:
                self.worst = ch
            if ch.score > self.best.score:
                self.best = ch
            if not self.flag and ch.flag:
                self.flag = True
        self.average = avr/self.number


def generate_init_population(pop_number: int, game: Game, score_mode: int = 0):
    pop = []
    for ch in range(0, pop_number):
        temp_str = ""
        for st in range(0, game.level_len):
            rand = random.random()
            if rand > 0.4:
                temp_str += '0'
            elif rand > 0.2:
                temp_str += '1'
            else:
                temp_str += '2'
        flag, score = game.get_score(temp_str, score_mode)
        pop.append(Chromosome(copy.deepcopy(temp_str), copy.deepcopy(flag), copy.deepcopy(score)))
    init_gen = Generation(pop)
    return init_gen


def selection(gen: Generation, mode: int = 0):
    selected = []
    if mode == 0:
        gen.members.sort(key=lambda x: x.score, reverse=True)
        selected = copy.deepcopy(gen.members[0: int(gen.number/2)])
    elif mode == 1:
        gen.members.sort(key=lambda x: x.score, reverse=True)
        min_score = gen.members[int(3 * gen.number / 4)].score
        chance_arr = []
        for ch in range(0, int(3 * gen.number / 4)):
            for j in range(0, int(gen.members[ch].score - min_score) + 1):
                chance_arr.append(ch)
        for b in range(0, int(gen.number / 2)):
            selected.append(copy.deepcopy(gen.members[random.choice(chance_arr)]))
    return selected


def crossover(selected: list, mode: int = 0):
    pop1 = copy.deepcopy(selected)
    pop2 = copy.deepcopy(selected)
    children = []
    parents = [None, None]
    for p in range(0, len(selected)):
        child1_str = ""
        child2_str = ""
        parents[0] = pop1.pop(random.randint(0, len(pop1)-1))
        parents[1] = pop2.pop(random.randint(0, len(pop2)-1))
        if mode == 0:
            for j in range(0, parents[0].len):
                choose = random.randint(0, 1)
                child1_str += parents[choose].string[j]
                child2_str += parents[1 - choose].string[j]
        elif mode == 1:
            child1_str += parents[0].string[0: int(parents[0].len / 2)]
            child1_str += parents[1].string[int(parents[1].len / 2): parents[1].len]
            child2_str += parents[1].string[0: int(parents[1].len / 2)]
            child2_str += parents[0].string[int(parents[0].len / 2): parents[0].len]
        elif mode == 2:
            child1_str += parents[0].string[0: int(parents[0].len / 3)]
            child1_str += parents[1].string[int(parents[1].len / 3): int(2 * parents[1].len / 3)]
            child1_str += parents[0].string[int(2 * parents[0].len / 3): parents[0].len]
            child2_str += parents[1].string[0: int(parents[1].len / 3)]
            child2_str += parents[0].string[int(parents[0].len / 3): int(2 * parents[0].len / 3)]
            child2_str += parents[1].string[int(2 * parents[1].len / 3): parents[1].len]
        children.append(Chromosome(copy.deepcopy(child1_str)))
        children.append(Chromosome(copy.deepcopy(child2_str)))
    return children


def mutation(children: list, p: float = 0.2):
    for ch in children:
        num = random.random()
        if num > 0.5:
            num = int(ch.len / 15)
        elif num > 0.2:
            num = int(ch.len / 10)
        else:
            num = int(ch.len / 5)

        which = random.sample(range(0, ch.len), num)
        temp_list = list(ch.string)
        for act in which:
            if random.random() < p:
                p = random.random()
                if p > 0.2:
                    m = '0'
                elif p > 0.1:
                    m = '1'
                else:
                    m = '2'
                temp_list[act] = m
        ch.string = copy.deepcopy("".join(temp_list))
    return children


def genetic_algorithm(game: Game, pop_number: int, s_mode: int = 0, c_mode: int = 0, score_mode: int = 0,
                      mutation_p: float = 0.2):
    init_gen = generate_init_population(pop_number, game, score_mode)
    generations = [init_gen]
    next_gen = init_gen
    prev_bst = -1000
    j = 0
    for gen in range(0, 500):
        selected_pop = selection(next_gen, s_mode)
        children_pop = crossover(selected_pop, c_mode)
        mutant_pop = mutation(children_pop, mutation_p)
        for ch in mutant_pop:
            ch.flag, ch.score = game.get_score(ch.string, score_mode)
        next_gen = Generation(mutant_pop)
        generations.append(next_gen)
        if 0.99 < next_gen.best.score/prev_bst < 1.01:
            j += 1
            if j == 30:
                return generations
        else:
            j = 0
        prev_bst = next_gen.best.score
    return generations


if __name__ == '__main__':
    test_read = open("levels/level8.txt", "r")
    test_write = open("outputs/level8_output.txt", "w")
    game1 = Game(test_read.readline())
    test_read.close()

    method_1 = [0, 0, 0, 0]
    method_2 = [0, 0, 0, 0]
    best_string_method1 = None
    best_string_method2 = None

    for i in range(0, 5):
        generations1 = genetic_algorithm(game1, 200, 0, 1, 0, 0.1)
        method_1[0] += len(generations1)
        method_1[1] += generations1[len(generations1) - 1].average
        method_1[2] += generations1[len(generations1) - 1].best.score
        n1 = np.arange(1, len(generations1) + 1, step=1)
        avr1 = []
        best1 = []
        worst1 = []
        first_flag1 = -10
        for gn in generations1:
            if gn.best.flag and (not best_string_method1 or best_string_method1.score < gn.best.score):
                best_string_method1 = gn.best
            avr1.append(gn.average)
            best1.append(gn.best.score)
            worst1.append(gn.worst.score)
            if first_flag1 == -10 and gn.flag:
                first_flag1 = generations1.index(gn) + 1
                method_1[3] += first_flag1
        style.use('ggplot')
        plt.plot(n1, avr1, label='Average')
        plt.plot(n1, best1, label='Best')
        plt.plot(n1, worst1, label='Worst')
        plt.stem(first_flag1, best1[first_flag1-1], 'g', markerfmt='go', label='First Winner')
        plt.legend()
        plt.show()

    for i in range(0, 5):
        generations2 = genetic_algorithm(game1, 500, 1, 2, 1, 0.5)
        method_2[0] += len(generations2)
        method_2[1] += generations2[len(generations2) - 1].average
        method_2[2] += generations2[len(generations2) - 1].best.score
        n2 = np.arange(1, len(generations2) + 1, step=1)
        avr2 = []
        best2 = []
        worst2 = []
        first_flag2 = -10
        for gn in generations2:
            if gn.best.flag and (not best_string_method2 or best_string_method2.score < gn.best.score):
                best_string_method2 = gn.best
            avr2.append(gn.average)
            best2.append(gn.best.score)
            worst2.append(gn.worst.score)
            if first_flag2 == -10 and gn.flag:
                first_flag2 = generations2.index(gn) + 1
                method_2[3] += first_flag2
        style.use('ggplot')
        plt.plot(n2, avr2, label='Average')
        plt.plot(n2, best2, label='Best')
        plt.plot(n2, worst2, label='Worst')
        plt.stem(first_flag2, best2[first_flag2 - 1], 'g', markerfmt='go', label='First Winner')
        plt.legend()
        plt.show()

    for i in range(0, 4):
        method_1[i] /= 5
        method_2[i] /= 5
    plt.stem(['Convergence', 'Average', 'Best', 'First Winner'], method_1, 'g', markerfmt='go', label='Method 1')
    plt.stem(['Convergence', 'Average', 'Best', 'First Winner'], method_2, 'b', markerfmt='bo', label='Method 2')
    plt.legend()
    plt.show()

    if best_string_method1:
        test_write.write("Method 1 output:  " + best_string_method1.string +
                         "   " + "{:.2f}".format(best_string_method1.score))
    else:
        test_write.write("Method 1 output:  " + "No winner found")
    if best_string_method2:
        test_write.write("\nMethod 2 output:  " + best_string_method2.string +
                         "   " + "{:.2f}".format(best_string_method2.score))
    else:
        test_write.write("\nMethod 2 output:  " + "No winner found")
    test_write.close()
