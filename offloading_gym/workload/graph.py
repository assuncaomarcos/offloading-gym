#!/usr/bin/env python
# -*- coding: utf-8 -*-
from networkx.drawing.nx_agraph import from_agraph
from networkx.utils import open_file

from offloading_gym.task_graph import TaskGraph
import random
import math


def parse_dot(path: str) -> TaskGraph:
    try:
        import pygraphviz
    except ImportError as err:
        raise ImportError(
            "parse() requires pygraphviz " "https://pygraphviz.github.io"
        ) from err

    gr = pygraphviz.AGraph(file=path)
    tg = from_agraph(gr, create_using=TaskGraph)
    gr.clear()
    return tg


# Default daggen DAG values obtained from the C implementation
NUM_TASKS = 100
FAT = 0.5
REGULAR = 0.9
DENSITY = 0.5
MIN_DATA = 2048
MAX_DATA = 11264
MIN_ALPHA = 0.0
MAX_ALPHA = 0.2
JUMP_SIZE = 1
CCR = 0

def _get_random_int_around(x, perc):
    r = -perc + (2 * perc * random.random())
    new_int = max(1, int(x * (1.0 + r / 100.00)))
    return new_int

def _generate_tasks(n_tasks: int, fat: float, regular: float):
    # Compute the number of tasks per level
    integral_part = math.modf(math.exp(fat * math.log(n_tasks)))
    n_tasks_per_level = int(integral_part[1])
    n_levels = 0
    total_n_tasks = 0

    while True:
        tmp = _get_random_int_around(n_tasks_per_level, 100.0 - 100.0 * regular)
        tmp = min(tmp, n_tasks - total_n_tasks)

        break


  # double integral_part;
  # double op =0;
  # int nb_levels=0;
  # int *nb_tasks=NULL;
  # int nb_tasks_per_level;
  # int total_nb_tasks=0;
  # int tmp;
  #
  # /* assign a number of tasks per level */
  # while (1) {
  #   tmp = getIntRandomNumberAround(nb_tasks_per_level, 100.00 - 100.0
  #       *config.regular);
  #   if (total_nb_tasks + tmp > n_tasks) {
  #     tmp = config.n - total_nb_tasks;
  #   }
  #   nb_tasks=(int*)realloc(nb_tasks, (nb_levels+1)*sizeof(int));
  #   nb_tasks[nb_levels++] = tmp;
  #   total_nb_tasks += tmp;
  #   if (total_nb_tasks >= config.n)
  #     break;
  # }

#   /* Put info in the dag structure */
#   dag->nb_levels=nb_levels;
#   dag->levels=(Task **)calloc(dag->nb_levels, sizeof(Task*));
#   dag->nb_tasks_per_level = nb_tasks;
#   for (i=0; i<dag->nb_levels; i++) {
#     dag->levels[i] = (Task *)calloc(dag->nb_tasks_per_level[i],
#         sizeof(Task));
#     for (j=0; j<dag->nb_tasks_per_level[i]; j++) {
#       dag->levels[i][j] = (Task)calloc(1, sizeof(struct _Task));
#       /** Task cost computation                **/
#       /** (1) pick a data size (in elements)   **/
#       /** (2) pick a complexity                **/
#       /** (3) add a factor for N_2 and N_LOG_N **/
#       /** (4) multiply (1) by (2) and by (3)   **/
#       /** Cost are in flops                    **/
#
#       dag->levels[i][j]->data_size = ((int) getRandomNumberBetween(
#           config.mindata, config.maxdata) / 1024) * 1024;
#
#       op = getRandomNumberBetween(64.0, 512.0);
#
#       if (!config.ccr) {
#         dag->levels[i][j]->complexity = ((int) getRandomNumberBetween(
#             config.mindata, config.maxdata) % 3 + 1);
#       } else {
#         dag->levels[i][j]->complexity = (int)config.ccr;
#       }
#
#       switch (dag->levels[i][j]->complexity) {
#       case N_2:
#         dag->levels[i][j]->cost = (op * pow(
#             dag->levels[i][j]->data_size, 2.0));
#         break;
#       case N_LOG_N:
#         dag->levels[i][j]->cost = (2 * op * pow(
#             dag->levels[i][j]->data_size, 2.0)
#         * (log(dag->levels[i][j]->data_size)/log(2.0)));
#         break;
#       case N_3:
#         dag->levels[i][j]->cost
#         = pow(dag->levels[i][j]->data_size, 3.0);
#         break;
#       case MIXED:
#         fprintf(stderr, "Modulo error in complexity function\n");
#         break;
#       }
#
#       dag->levels[i][j]->alpha = getRandomNumberBetween(config.minalpha,
#           config.maxalpha);
#     }
#   }
# }


def daggen_graph(
        seed: int = None,
        num_tasks: int = NUM_TASKS,
        min_data: int = MIN_DATA,
        max_data: int = MAX_DATA,
        min_alpha: float = MIN_ALPHA,
        max_alpha: float = MAX_ALPHA,
        fat: float = FAT,
        density: float = DENSITY,
        regular: float = REGULAR,
        ccr: float = CCR,
        jump_size: int = JUMP_SIZE
) -> TaskGraph:
    assert num_tasks >= 1, "num_tasks must be >= 1"
    assert 0 <= min_data <= max_data <= 1, ("min_data and max_data must be between 0..1, "
                                            "min_data must be smaller than max_data")
    assert 0 <= min_alpha <= max_alpha <= 1, ("min_alpha and max_alpha must be between 0..1, "
                                              "min_alpha must be smaller than max_alpha")
    assert 0 <= fat <= 1, "fat must be between 0 and 1"
    assert 0 <= density <= 1, "density must be between 0 and 1"
    assert 0 <= regular <= 1, "regular must be between 0 and 1"
    assert 0 <= ccr <= 3, "ccr must be between 0 and 3"

    if seed is None:
        seed = random.randint(0, 99999999)
    random.seed(seed)

    graph = TaskGraph()

    return graph


