# %%
import os
import getpass
# %%
WINDOWS_LOG_ROOT = os.path.join(
    'C:/',
    'users',
    getpass.getuser(),
    'AppData',
    'LocalLow',
)

LOG_INTERMEDIATE = os.path.join('Wizards Of The Coast', 'MTGA')
LOG_PATH = os.path.join(WINDOWS_LOG_ROOT,LOG_INTERMEDIATE)
# %%
