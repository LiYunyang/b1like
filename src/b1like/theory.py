# follow example in /sdf/home/w/wlwu/repos/2025/202503_cosmopower/cobaya_theory_cpj.py

from cobaya.theories.cosmo import BoltzmannBase
from cobaya.typing import InfoDict

from importlib import resources
import os
import numpy as np


def get_BKsim_spec(fname):
    """
    Load the BK simulation spectra.

    The lensing BB spectrum should match the SPT-3G simulations. The
    BK-generated tensor ``r=1`` spectrum is used for tensor BB scaling.
    """
    return np.load(fname)  # tmp['cmbt'] gives r=1 bb spec; tmp['cmbl'] gives lensing bb spec


class FixedLCDM(BoltzmannBase):
    """
    Theory class with fixed lensing and tensor BB spectra.

    The spectra can be scaled with the ``Al_scale`` and ``r`` parameters. The
    fixed spectra come from the inputs used to generate the BK simulations, and
    the lensed-LCDM spectrum should match the input to SPT-3G.

    """

    fixlcdm_fname = None

    extra_args: InfoDict = {}

    def initialize(self):
        super().initialize()

        # have all objects, when relevant, start at ell=1 (bpwf, theory spec)

        if self.fixlcdm_fname is None:
            spec_path = resources.files("b1like").joinpath("fixlcdm_spec.npz")
        else:
            spec_path = self.fixlcdm_fname

        if not os.path.isfile(spec_path):
            raise FileNotFoundError(f"Fixed LCDM spectrum file not found: {spec_path}")

        spec = get_BKsim_spec(spec_path)
        tens_bb = spec['cmbt']
        lens_bb = spec['cmbl']

        self.lmax_theory = 999  # self.extra_args.get("lmax_theory", 999)

        self.tens_bb = np.concatenate([tens_bb, np.zeros(self.lmax_theory - len(tens_bb) + 1)])
        self.lens_bb = lens_bb[: self.lmax_theory + 1]

        self.r_parname = "r"
        self.Al_parname = "Al_scale"

        ells = np.arange(0, self.lmax_theory + 1)
        ells[0] = 1  # avoid divide by zero
        self.ells = ells

    def initialize_with_provider(self, provider):
        """
        Initialize with the provider after other components are initialized.

        The provider class instance is used to return dependencies; see
        ``calculate`` below.
        """
        self.provider = provider

    def get_requirements(self):
        """
        Return required derived parameters and quantities.

        The returned quantities are needed by this component and should be
        calculated by another theory class.
        """
        return {self.r_parname: None, self.Al_parname: None}

    def must_provide(self, **requirements):
        """Return dictionary of parameters that must be provided."""
        return {self.r_parname: None, self.Al_parname: None}

    def get_can_provide(self):
        """Return list of quantities that can be provided."""
        return ["Dl"]

    def calculate(self, state, want_derived=True, **params):
        """Return total ``Cl``/``Dl`` and lensed-scalar ``Cl``/``Dl``."""
        ellfac = self.ells * (self.ells + 1) / 2 / np.pi

        state["Dl"] = {
            "ell": self.ells,
            "bb": state['params'][self.r_parname] * self.tens_bb
            + state['params'][self.Al_parname] * self.lens_bb,
        }

        state["Cl"] = {"ell": self.ells, "bb": state["Dl"]["bb"] / ellfac}

        state["Dl_lens"] = {"ell": self.ells, "bb": state['params'][self.Al_parname] * self.lens_bb}

        state["Cl_lens"] = {"ell": self.ells, "bb": state["Dl_lens"]["bb"] / ellfac}

        return

    def get_Cl(self, ell_factor=False, **kwargs):
        """Return the total tensor-plus-lensed-scalar ``Cl`` values or ``Dl`` values."""
        if ell_factor:
            return self.current_state["Dl"].copy()
        else:
            return self.current_state["Cl"].copy()

    def get_Cl_lensed_scalar(self, ell_factor=False, **kwargs):
        """Return the lensed-scalar ``Cl`` values or ``Dl`` values."""
        if ell_factor:
            return self.current_state["Dl_lens"].copy()
        else:
            return self.current_state["Cl_lens"].copy()
