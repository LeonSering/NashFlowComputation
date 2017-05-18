# ===========================================================================
# Author:       Max ZIMMER
# Project:      NashFlowComputation 2017
# File:         utilitiesClass.py
# Description:
# ===========================================================================

import os
import time
class Utilities:

    @staticmethod
    def create_dir(path):
        if not os.path.isdir(path):
            os.mkdir(path)

    @staticmethod
    def get_time():
        return time.strftime("%d_%m_%Y-%H_%M_%S", time.localtime())