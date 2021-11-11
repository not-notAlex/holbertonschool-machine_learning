#!/usr/bin/env python3
"""
moduel for task 6
"""

import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    performs Baum-Welch algorithm for HMM
    """
    
