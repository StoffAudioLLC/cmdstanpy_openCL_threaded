# pylint: disable=wrong-import-position
"""CmdStanPy Module"""

import atexit
import shutil
import tempfile

_STANSUMMARY_STATS = [
    'Mean',
    'MCSE',
    'StdDev',
    '5%',
    '50%',
    '95%',
    'N_Eff',
    'N_Eff/s',
    'R_hat',
]

_TMPDIR = tempfile.mkdtemp()
_CMDSTAN_WARMUP = 1000
_CMDSTAN_SAMPLING = 1000
_CMDSTAN_THIN = 1
_DOT_CMDSTANPY = '.cmdstanpy'
_DOT_CMDSTAN = '.cmdstan'


def _cleanup_tmpdir():
    """Force deletion of _TMPDIR."""
    print('deleting tmpfiles dir: {}'.format(_TMPDIR))
    #shutil.rmtree(_TMPDIR, ignore_errors=True)
    print('done')


atexit.register(_cleanup_tmpdir)


from ._version import __version__  # noqa
from .compilation import compile_stan_file
from .install_cmdstan import rebuild_cmdstan
from .model import CmdStanModel
from .stanfit import (
    CmdStanGQ,
    CmdStanLaplace,
    CmdStanMCMC,
    CmdStanMLE,
    CmdStanPathfinder,
    CmdStanVB,
    from_csv,
)
from .utils import (
    cmdstan_path,
    cmdstan_version,
    install_cmdstan,
    set_cmdstan_path,
    set_make_env,
    show_versions,
    write_stan_json,
)

__all__ = [
    'set_cmdstan_path',
    'cmdstan_path',
    'set_make_env',
    'install_cmdstan',
    'compile_stan_file',
    'CmdStanMCMC',
    'CmdStanMLE',
    'CmdStanGQ',
    'CmdStanVB',
    'CmdStanLaplace',
    'CmdStanPathfinder',
    'CmdStanModel',
    'from_csv',
    'write_stan_json',
    'show_versions',
    'rebuild_cmdstan',
    'cmdstan_version',
]
