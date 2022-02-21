import math
import matplotlib.pyplot as plt
import numpy as np
import preprocess
import random
from tqdm import tqdm
import time

class Support_Vector_Machine:
    def __init__(self, points):
        self.points = points
