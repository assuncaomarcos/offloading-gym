#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module downloads the Wonderproxy dataset and extracts the latency data
required to model the latency between datacenters.

For more details on the dataset, check
`here <https://wonderproxy.com/blog/a-day-in-the-life-of-the-internet/>`_.
"""

from .wonderproxy import main

if __name__ == "__main__":
    main()
    