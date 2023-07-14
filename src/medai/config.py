# Copyright (C) 2023 Elif Cansu YILDIZ
# 
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or (at your option) any later
# version.
# 
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
# 
# You should have received a copy of the GNU General Public License along with
# this program; if not, see <http://www.gnu.org/licenses/>.

from pathlib import Path
from os.path import dirname, abspath
from functools import partial
from tqdm import tqdm
import coloredlogs
import logging

# Directories
BASE_DIR = Path(__file__).parent.parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "configs")
DATA_DIR = Path(BASE_DIR, "datasets")

# make all tqdm pbars dynamic to fit any window resolution
tqdm = partial(tqdm, dynamic_ncols=True)

coloredlogs.install(level=logging.INFO)
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def example_loggers():
    # Sample messages (note that we use configured `logger` now)
    logger.debug("Used for debugging your code.")
    logger.info("Informative messages from your code.")
    logger.warning("Everything works but there is something to be aware of.")
    logger.error("There's been a mistake with the process.")
    logger.critical("There is something terribly wrong and process may terminate.")