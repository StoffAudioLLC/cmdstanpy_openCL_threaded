#!/usr/bin/env python
"""
Download and install a CmdStan release from GitHub.
Downloads the release tar.gz file to temporary storage.
Retries GitHub requests in order to allow for transient network outages.
Builds CmdStan executables and tests the compiler by building
example model ``bernoulli.stan``.

Optional command line arguments:
   -v, --version <release> : version, defaults to latest release version
   -d, --dir <path> : install directory, defaults to '$HOME/.cmdstan(py)
   --overwrite: flag, when specified re-installs existing version
   --verbose: flag, when specified prints output from CmdStan build process
   --progress: flag, when specified show progress bar for CmdStan download

   -c, --compiler : flag, add C++ compiler to path (Windows only)
"""
import argparse
import contextlib
import json
import os
import platform
import re
import subprocess
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
from collections import OrderedDict
from pathlib import Path
from time import sleep

from cmdstanpy import _DOT_CMDSTAN, _DOT_CMDSTANPY
from cmdstanpy.utils import validate_dir

EXTENSION = '.exe' if platform.system() == 'Windows' else ''


class CmdStanRetrieveError(RuntimeError):
    pass


class CmdStanInstallError(RuntimeError):
    pass


@contextlib.contextmanager
def pushd(new_dir: str):
    """Acts like pushd/popd."""
    previous_dir = os.getcwd()
    os.chdir(new_dir)
    yield
    os.chdir(previous_dir)


def usage():
    """Print usage."""
    msg = """
    Arguments:
        -v (--version) : CmdStan version
        -d (--dir) : install directory
        --overwrite : replace installed version
        --verbose : show CmdStan build messages
        --progress : show progress bar for CmdStan download
        """

    if platform.system() == "Windows":
        msg += "-c (--compiler) : add C++ compiler to path (Windows only)\n"

    msg += "        -h (--help) : this message"

    print(msg)


def install_version(
    cmdstan_version: str, overwrite: bool = False, verbose: bool = False
):
    """
    Build specified CmdStan version by spawning subprocesses to
    run the Make utility on the downloaded CmdStan release src files.
    Assumes that current working directory is parent of release dir.

    :param cmdstan_version: CmdStan release, corresponds to release dirname.
    :param overwrite: when ``True``, run ``make clean-all`` before building.
    :param verbose: when ``True``, print build msgs to stdout.
    """
    with pushd(cmdstan_version):
        make = os.getenv(
            'MAKE', 'make' if platform.system() != 'Windows' else 'mingw32-make'
        )
        print('Building version {}'.format(cmdstan_version))
        if overwrite:
            print(
                'Overwrite requested, remove existing build of version '
                '{}'.format(cmdstan_version)
            )
            cmd = [make, 'clean-all']
            proc = subprocess.Popen(
                cmd,
                cwd=None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=os.environ,
            )
            while proc.poll() is None:
                output = proc.stdout.readline().decode('utf-8').strip()
                if verbose and output:
                    print(output, flush=True)
            _, stderr = proc.communicate()
            if proc.returncode:
                msgs = ['Command "make clean-all" failed']
                if stderr:
                    msgs.append(stderr.decode('utf-8').strip())
                raise CmdStanInstallError('\n'.join(msgs))
            print('Rebuilding version {}'.format(cmdstan_version))
        cmd = [make, 'build']
        proc = subprocess.Popen(
            cmd,
            cwd=None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ,
        )
        while proc.poll() is None:
            output = proc.stdout.readline().decode('utf-8').strip()
            if verbose and output:
                print(output, flush=True)
        _, stderr = proc.communicate()
        if proc.returncode:
            msgs = ['Command "make build" failed']
            if stderr:
                msgs.append(stderr.decode('utf-8').strip())
            raise CmdStanInstallError('\n'.join(msgs))
        print('Test model compilation')
        cmd = [
            make,
            Path(
                os.path.join('examples', 'bernoulli', 'bernoulli' + EXTENSION)
            ).as_posix(),
        ]
        if platform.system() == "Windows":
            # Add tbb to the $PATH on Windows
            libtbb = os.path.join(
                os.getcwd(), 'stan', 'lib', 'stan_math', 'lib', 'tbb'
            )
            os.environ['PATH'] = ';'.join(
                list(
                    OrderedDict.fromkeys(
                        [libtbb] + os.environ.get('PATH', '').split(';')
                    )
                )
            )
        proc = subprocess.Popen(
            cmd,
            cwd=None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ,
        )
        while proc.poll() is None:
            proc.stdout.readline().decode('utf-8')
        _, stderr = proc.communicate()
        if proc.returncode:
            msgs = ['Failed to compile example model bernoulli.stan']
            if stderr:
                msgs.append(stderr.decode('utf-8').strip())
            raise CmdStanInstallError('\n'.join(msgs))
    print('Installed {}'.format(cmdstan_version))


def is_version_available(version: str):
    is_available = True
    url = (
        'https://github.com/stan-dev/cmdstan/releases/download/'
        'v{0}/cmdstan-{0}.tar.gz'.format(version)
    )
    for i in range(6):
        try:
            urllib.request.urlopen(url)
        except urllib.error.HTTPError as err:
            print('Release {} is unavailable from URL {}'.format(version, url))
            print('HTTPError: {}'.format(err.code))
            is_available = False
            break
        except urllib.error.URLError as e:
            if i < 5:
                print(
                    'checking version {} availability, retry ({}/5)'.format(
                        version, i + 1
                    )
                )
                sleep(1)
                continue
            print('Release {} is unavailable from URL {}'.format(version, url))
            print('URLError: {}'.format(e.reason))
            is_available = False
    return is_available


def get_headers():
    """Create headers dictionary."""
    headers = {}
    GITHUB_PAT = os.environ.get("GITHUB_PAT")  # pylint:disable=invalid-name
    if GITHUB_PAT is not None:
        headers["Authorization"] = "token {}".format(GITHUB_PAT)
    return headers


def latest_version():
    """Report latest CmdStan release version."""
    url = 'https://api.github.com/repos/stan-dev/cmdstan/releases/latest'
    request = urllib.request.Request(url, headers=get_headers())
    for i in range(6):
        try:
            response = urllib.request.urlopen(request).read()
            break
        except urllib.error.URLError as e:
            print('Cannot connect to github.')
            print(e)
            if i < 5:
                print('retry ({}/5)'.format(i + 1))
                sleep(1)
                continue
            raise CmdStanRetrieveError(
                'Cannot connect to CmdStan github repo.'
            ) from e
    content = json.loads(response.decode('utf-8'))
    tag = content['tag_name']
    match = re.search(r'v?(.+)', tag)
    if match is not None:
        tag = match.group(1)
    return tag  # type: ignore


def home_cmdstan() -> str:
    return os.path.expanduser(os.path.join('~', _DOT_CMDSTAN))


# pylint: disable=too-few-public-methods
class InstallationSettings:
    """
    A static installation settings object
    """

    def __init__(
        self,
        *,
        version: Optional[str] = None,
        dir: Optional[str] = None,
        progress: bool = False,
        verbose: bool = False,
        overwrite: bool = False,
        cores: int = 1,
        compiler: bool = False,
        **kwargs: Any,
    ):
        self.version = version if version else latest_version()
        self.dir = dir if dir else home_cmdstan()
        self.progress = progress
        self.verbose = verbose
        self.overwrite = overwrite
        self.cores = cores
        self.compiler = compiler and is_windows()

        _ = kwargs  # ignore all other inputs.
        # Useful if initialized from a dictionary like **dict


def yes_no(answer: str, default: bool) -> bool:
    answer = answer.lower()
    if answer in ('y', 'yes'):
        return True
    if answer in ('n', 'no'):
        return False
    return default


class InteractiveSettings:
    """
    Installation settings provided on-demand in an interactive format.

    This provides the same set of properties as the ``InstallationSettings``
    object, but rather than them being fixed by the constructor the user is
    asked for input whenever they are accessed for the first time.
    """

    @cached_property
    def version(self) -> str:
        latest = latest_version()
        print("Which version would you like to install?")
        print(f"Default: {latest}")
        answer = input("Type version or hit enter to continue: ")
        return answer if answer else latest

    @cached_property
    def dir(self) -> str:
        directory = home_cmdstan()
        print("Where would you like to install CmdStan?")
        print(f"Default: {directory}")
        answer = input("Type full path or hit enter to continue: ")
        return os.path.expanduser(answer) if answer else directory

    @cached_property
    def progress(self) -> bool:
        print("Show installation progress bars?")
        print("Default: y")
        answer = input("[y/n]: ")
        return yes_no(answer, True)

    @cached_property
    def verbose(self) -> bool:
        print("Show verbose output of the installation process?")
        print("Default: n")
        answer = input("[y/n]: ")
        return yes_no(answer, False)

    @cached_property
    def overwrite(self) -> bool:
        print("Overwrite existing CmdStan installation?")
        print("Default: n")
        answer = input("[y/n]: ")
        return yes_no(answer, False)

    @cached_property
    def compiler(self) -> bool:
        if not is_windows():
            return False
        print("Would you like to install the RTools40 C++ toolchain?")
        print("A C++ toolchain is required for CmdStan.")
        print(
            "If you are not sure if you need the toolchain or not, "
            "the most likely case is you do need it, and should answer 'y'."
        )
        print("Default: n")
        answer = input("[y/n]: ")
        return yes_no(answer, False)

    @cached_property
    def cores(self) -> int:
        max_cpus = os.cpu_count() or 1
        print(
            "How many CPU cores would you like to use for installing "
            "and compiling CmdStan?"
        )
        print(f"Default: 1, Max: {max_cpus}")
        answer = input("Enter a number or hit enter to continue: ")
        try:
            return min(max_cpus, max(int(answer), 1))
        except ValueError:
            return 1


def clean_all(verbose: bool = False) -> None:
    """
    Run `make clean-all` in the current directory (must be a cmdstan library).

    :param verbose: Boolean value; when ``True``, show output from make command.
    """
    cmd = [MAKE, 'clean-all']
    try:
        if verbose:
            do_command(cmd)
        else:
            do_command(cmd, fd_out=None)

    except RuntimeError as e:
        # pylint: disable=raise-missing-from
        raise CmdStanInstallError(f'Command "make clean-all" failed\n{str(e)}')


def build(verbose: bool = False, progress: bool = True, cores: int = 1) -> None:
    """
    Run command ``make build`` in the current directory, which must be
    the home directory of a CmdStan version (or GitHub repo).
    By default, displays a progress bar which tracks make command outputs.
    If argument ``verbose=True``, instead of a progress bar, streams
    make command outputs to sys.stdout.  When both ``verbose`` and ``progress``
    are ``False``, runs silently.

    :param verbose: Boolean value; when ``True``, show output from make command.
        Default is ``False``.
    :param progress: Boolean value; when ``True`` display progress progress bar.
        Default is ``True``.
    :param cores: Integer, number of cores to use in the ``make`` command.
        Default is 1 core.
    """
    cmd = [MAKE, 'build', f'-j{cores}']
    try:
        if verbose:
            do_command(cmd)
        elif progress and progbar.allow_show_progress():
            progress_hook: Any = _wrap_build_progress_hook()
            do_command(cmd, fd_out=None, pbar=progress_hook)
        else:
            do_command(cmd, fd_out=None)

    except RuntimeError as e:
        # pylint: disable=raise-missing-from
        raise CmdStanInstallError(f'Command "make build" failed\n{str(e)}')
    if not os.path.exists(os.path.join('bin', 'stansummary' + EXTENSION)):
        raise CmdStanInstallError(
            f'bin/stansummary{EXTENSION} not found'
            ', please rebuild or report a bug!'
        )
    if not os.path.exists(os.path.join('bin', 'diagnose' + EXTENSION)):
        raise CmdStanInstallError(
            f'bin/stansummary{EXTENSION} not found'
            ', please rebuild or report a bug!'
        )

    if is_windows():
        # Add tbb to the $PATH on Windows
        libtbb = os.path.join(
            os.getcwd(), 'stan', 'lib', 'stan_math', 'lib', 'tbb'
        )
        os.environ['PATH'] = ';'.join(
            list(
                OrderedDict.fromkeys(
                    [libtbb] + os.environ.get('PATH', '').split(';')
                )
            )
        )


@progbar.wrap_callback
def _wrap_build_progress_hook() -> Optional[Callable[[str], None]]:
    """Sets up tqdm callback for CmdStan sampler console msgs."""
    pad = ' ' * 20
    msgs_expected = 150  # hack: 2.27 make build send ~140 msgs to console
    pbar: tqdm = tqdm(
        total=msgs_expected,
        bar_format="{desc} ({elapsed}) | {bar} | {postfix[0][value]}",
        postfix=[{"value": f'Building CmdStan {pad}'}],
        colour='blue',
        desc='',
        position=0,
    )

    def build_progress_hook(line: str) -> None:
        if line.startswith('--- CmdStan'):
            pbar.set_description('Done')
            pbar.postfix[0]["value"] = line
            pbar.update(msgs_expected - pbar.n)
            pbar.close()
        else:
            if line.startswith('--'):
                pbar.postfix[0]["value"] = line
            else:
                pbar.postfix[0]["value"] = f'{line[:8]} ... {line[-20:]}'
                pbar.set_description('Compiling')
                pbar.update(1)

    return build_progress_hook


def compile_example(verbose: bool = False) -> None:
    """
    Compile the example model.
    The current directory must be a cmdstan installation, i.e.,
    contains the makefile, Stanc compiler, and all libraries.

    :param verbose: Boolean value; when ``True``, show output from make command.
    """
    path = Path('examples', 'bernoulli', 'bernoulli').with_suffix(EXTENSION)
    if path.is_file():
        path.unlink()

    cmd = [MAKE, path.as_posix()]
    try:
        if verbose:
            do_command(cmd)
        else:
            do_command(cmd, fd_out=None)
    except RuntimeError as e:
        # pylint: disable=raise-missing-from
        raise CmdStanInstallError(f'Command "{" ".join(cmd)}" failed:\n{e}')

    if not path.is_file():
        raise CmdStanInstallError("Failed to generate example binary")


def rebuild_cmdstan(
    verbose: bool = False, progress: bool = True, cores: int = 1
) -> None:
    """
    Rebuilds the existing CmdStan installation.
    This assumes CmdStan has already been installed,
    though it need not be installed via CmdStanPy for
    this function to work.

    :param verbose: Boolean value; when ``True``, show output from make command.
        Default is ``False``.
    :param progress: Boolean value; when ``True`` display progress progress bar.
        Default is ``True``.
    :param cores: Integer, number of cores to use in the ``make`` command.
        Default is 1 core.
    """
    try:
        with pushd(cmdstan_path()):
            clean_all(verbose)
            build(verbose, progress, cores)
            compile_example(verbose)
    except ValueError as e:
        raise CmdStanInstallError(
            "Failed to rebuild CmdStan. Are you sure it is installed?"
        ) from e


def install_version(
    cmdstan_version: str,
    overwrite: bool = False,
    verbose: bool = False,
    progress: bool = True,
    cores: int = 1,
) -> None:
    """
    Build specified CmdStan version by spawning subprocesses to
    run the Make utility on the downloaded CmdStan release src files.
    Assumes that current working directory is parent of release dir.

    :param cmdstan_version: CmdStan release, corresponds to release dirname.
    :param overwrite: when ``True``, run ``make clean-all`` before building.
    :param verbose: Boolean value; when ``True``, show output from make command.
    """
    with pushd(cmdstan_version):
        print(
            'Building version {}, may take several minutes, '
            'depending on your system.'.format(cmdstan_version)
        )
        if overwrite and os.path.exists('.'):
            print(
                'Overwrite requested, remove existing build of version '
                '{}'.format(cmdstan_version)
            )
            clean_all(verbose)
            print('Rebuilding version {}'.format(cmdstan_version))
        build(verbose, progress=progress, cores=cores)
    print('Installed {}'.format(cmdstan_version))


def is_version_available(version: str) -> bool:
    if 'git:' in version:
        return True  # no good way in general to check if a git tag exists

    is_available = True
    url = get_download_url(version)
    for i in range(6):
        try:
            urllib.request.urlopen(url)
        except urllib.error.HTTPError as err:
            print(f'Release {version} is unavailable from URL {url}')
            print(f'HTTPError: {err.code}')
            is_available = False
            break
        except urllib.error.URLError as e:
            if i < 5:
                print(
                    'checking version {} availability, retry ({}/5)'.format(
                        version, i + 1
                    )
                )
                sleep(1)
                continue
            print('Release {} is unavailable from URL {}'.format(version, url))
            print('URLError: {}'.format(e.reason))
            is_available = False
    return is_available


def retrieve_version(version: str, progress: bool = True) -> None:
    """Download specified CmdStan version."""
    if version is None or version == '':
        raise ValueError('Argument "version" unspecified.')

    if 'git:' in version:
        tag = version.split(':')[1]
        tag_folder = version.replace(':', '-').replace('/', '_')
        print(f"Cloning CmdStan branch '{tag}' from stan-dev/cmdstan on GitHub")
        do_command(
            [
                'git',
                'clone',
                '--depth',
                '1',
                '--branch',
                tag,
                '--recursive',
                '--shallow-submodules',
                'https://github.com/stan-dev/cmdstan.git',
                f'cmdstan-{tag_folder}',
            ]
        )
        return

    print('Downloading CmdStan version {}'.format(version))
    url = (
        'https://github.com/stan-dev/cmdstan/releases/download/'
        'v{0}/cmdstan-{0}.tar.gz'.format(version)
    )
    for i in range(6):  # always retry to allow for transient URLErrors
        try:
            if progress:
                progress_hook = wrap_progress_hook()
            else:
                progress_hook = None
            file_tmp, _ = urllib.request.urlretrieve(
                url, filename=None, reporthook=progress_hook
            )
            break
        except urllib.error.HTTPError as e:
            raise CmdStanRetrieveError(
                'HTTPError: {}\n'
                'Version {} not available from github.com.'.format(
                    e.code, version
                )
            ) from e
        except urllib.error.URLError as e:
            print(
                'Failed to download CmdStan version {} from github.com'.format(
                    version
                )
            )
            print(e)
            if i < 5:
                print('retry ({}/5)'.format(i + 1))
                sleep(1)
                continue
            print('Version {} not available from github.com.'.format(version))
            raise CmdStanRetrieveError(
                'Version {} not available from github.com.'.format(version)
            ) from e
    print('Download successful, file: {}'.format(file_tmp))
    try:
        tar = tarfile.open(file_tmp)
        target = os.getcwd()
        if platform.system() == 'Windows':
            # fixes long-path limitation on Windows
            target = r'\\?\{}'.format(target)
        tar.extractall(target)
    except Exception as e:  # pylint: disable=broad-except
        raise CmdStanInstallError(
            'Failed to unpack file {}'.format(file_tmp)
        ) from e
    finally:
        tar.close()
    print(f'Unpacked download as {cmdstan_dir}')


def run_compiler_install(dir: str, verbose: bool, progress: bool) -> None:
    from .install_cxx_toolchain import is_installed as _is_installed_cxx
    from .install_cxx_toolchain import run_rtools_install as _main_cxx
    from .utils import cxx_toolchain_path

    compiler_found = False
    rtools40_home = os.environ.get('RTOOLS40_HOME')
    for cxx_loc in ([rtools40_home] if rtools40_home is not None else []) + [
        home_cmdstan(),
        os.path.join(os.path.abspath("/"), "RTools40"),
        os.path.join(os.path.abspath("/"), "RTools"),
        os.path.join(os.path.abspath("/"), "RTools35"),
        os.path.join(os.path.abspath("/"), "RBuildTools"),
    ]:
        for cxx_version in ['40', '35']:
            if _is_installed_cxx(cxx_loc, cxx_version):
                compiler_found = True
                break
        if compiler_found:
            break
    if not compiler_found:
        print('Installing RTools40')
        # copy argv and clear sys.argv
        _main_cxx(
            {
                'dir': dir,
                'progress': progress,
                'version': None,
                'verbose': verbose,
            }
        )
        cxx_version = '40'
    # Add toolchain to $PATH
    cxx_toolchain_path(cxx_version, dir)


def run_install(args: Union[InteractiveSettings, InstallationSettings]) -> None:
    """
    Run a (potentially interactive) installation
    """
    validate_dir(args.dir)
    print('CmdStan install directory: {}'.format(args.dir))

    # these accesses just 'warm up' the interactive install
    _ = args.progress
    _ = args.verbose

    if args.compiler:
        run_compiler_install(args.dir, args.verbose, args.progress)

    if 'git:' in args.version:
        tag = args.version.replace(':', '-').replace('/', '_')
        cmdstan_version = f'cmdstan-{tag}'
    else:
        cmdstan_version = f'cmdstan-{args.version}'
    with pushd(args.dir):
        already_installed = os.path.exists(cmdstan_version) and os.path.exists(
            os.path.join(
                cmdstan_version,
                'examples',
                'bernoulli',
                'bernoulli' + EXTENSION,
            )
        )
        if not already_installed or args.overwrite:
            if is_version_available(args.version):
                print('Installing CmdStan version: {}'.format(args.version))
            else:
                raise ValueError(
                    f'Version {args.version} cannot be downloaded. '
                    'Connection to GitHub failed. '
                    'Check firewall settings or ensure this version exists.'
                )
            shutil.rmtree(cmdstan_version, ignore_errors=True)
            retrieve_version(args.version, args.progress)
            install_version(
                cmdstan_version=cmdstan_version,
                overwrite=already_installed and args.overwrite,
                verbose=args.verbose,
                progress=args.progress,
                cores=args.cores,
            )
        else:
            print('CmdStan version {} already installed'.format(args.version))

        with pushd(cmdstan_version):
            print('Test model compilation')
            compile_example(args.verbose)


def parse_cmdline_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser("install_cmdstan")
    parser.add_argument(
        '--interactive',
        '-i',
        action='store_true',
        help="Ignore other arguments and run the installation in "
        + "interactive mode",
    )
    parser.add_argument(
        '--version',
        '-v',
        help="version, defaults to latest release version. "
        "If git is installed, you can also specify a git tag or branch, "
        "e.g. git:develop",
    )
    parser.add_argument(
        '--dir', '-d', help="install directory, defaults to '$HOME/.cmdstan(py)"
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help="flag, when specified re-installs existing version",
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help="flag, when specified prints output from CmdStan build process",
    )
    parser.add_argument(
        '--progress',
        action='store_true',
        help="flag, when specified show progress bar for CmdStan download",
    )
    if platform.system() == 'Windows':
        # use compiler installed with install_cxx_toolchain
        # Install a new compiler if compiler not found
        # Search order is RTools40, RTools35
        parser.add_argument(
            '--compiler',
            '-c',
            dest='compiler',
            action='store_true',
            help="flag, add C++ compiler to path (Windows only)",
        )
    args = parser.parse_args(sys.argv[1:])

    version = latest_version()
    if vars(args)['version']:
        version = vars(args)['version']

    if is_version_available(version):
        print('Installing CmdStan version: {}'.format(version))
    else:
        raise ValueError(
            'Invalid version requested: {}, cannot install.'.format(version)
        )

    cmdstan_dir = os.path.expanduser(os.path.join('~', _DOT_CMDSTAN))
    if not os.path.exists(cmdstan_dir):
        cmdstanpy_dir = os.path.expanduser(os.path.join('~', _DOT_CMDSTANPY))
        if os.path.exists(cmdstanpy_dir):
            cmdstan_dir = cmdstanpy_dir

    install_dir = cmdstan_dir
    if vars(args)['dir']:
        install_dir = vars(args)['dir']

    validate_dir(install_dir)
    print('Install directory: {}'.format(install_dir))

    if vars(args)['progress']:
        progress = vars(args)['progress']
        try:
            # pylint: disable=unused-import
            from tqdm import tqdm  # noqa: F401
        except (ImportError, ModuleNotFoundError):
            progress = False
    else:
        progress = False

    if platform.system() == 'Windows' and vars(args)['compiler']:
        from .install_cxx_toolchain import is_installed as _is_installed_cxx
        from .install_cxx_toolchain import main as _main_cxx
        from .utils import cxx_toolchain_path

        cxx_loc = cmdstan_dir
        compiler_found = False
        rtools40_home = os.environ.get('RTOOLS40_HOME')
        for cxx_loc in (
            [rtools40_home] if rtools40_home is not None else []
        ) + [
            cmdstan_dir,
            os.path.join(os.path.abspath("/"), "RTools40"),
            os.path.join(os.path.abspath("/"), "RTools"),
            os.path.join(os.path.abspath("/"), "RTools35"),
            os.path.join(os.path.abspath("/"), "RBuildTools"),
        ]:
            for cxx_version in ['40', '35']:
                if _is_installed_cxx(cxx_loc, cxx_version):
                    compiler_found = True
                    break
            if compiler_found:
                break
        if not compiler_found:
            print('Installing RTools40')
            # copy argv and clear sys.argv
            original_argv = sys.argv[:]
            sys.argv = [
                item for item in sys.argv if item not in ("--compiler", "-c")
            ]
            _main_cxx()
            sys.argv = original_argv
            cxx_version = '40'
        # Add toolchain to $PATH
        cxx_toolchain_path(cxx_version)

    cmdstan_version = 'cmdstan-{}'.format(version)
    with pushd(install_dir):
        if vars(args)['overwrite'] or not (
            os.path.exists(cmdstan_version)
            and os.path.exists(
                os.path.join(
                    cmdstan_version,
                    'examples',
                    'bernoulli',
                    'bernoulli' + EXTENSION,
                )
            )
        ):
            try:
                retrieve_version(version, progress)
                install_version(
                    cmdstan_version=cmdstan_version,
                    overwrite=vars(args)['overwrite'],
                    verbose=vars(args)['verbose'],
                )
            except RuntimeError as e:
                print(e)
                sys.exit(3)
        else:
            print('CmdStan version {} already installed'.format(version))


if __name__ == '__main__':
    main()
