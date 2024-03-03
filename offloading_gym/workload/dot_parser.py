#!/usr/bin/env python
# -*- coding: utf-8 -*-

from networkx import DiGraph
from networkx.drawing.nx_agraph import from_agraph


def parse_dot(path: str) -> DiGraph:
    try:
        import pygraphviz
    except ImportError as err:
        raise ImportError(
            "parse() requires pygraphviz " "https://pygraphviz.github.io"
        ) from err

    gr = pygraphviz.AGraph(file=path)
    tg = from_agraph(gr, create_using=DiGraph)
    gr.clear()
    return tg
