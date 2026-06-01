# follow example in /sdf/home/w/wlwu/repos/2025/202503_cosmopower/cobaya_theory_cpj.py

from cobaya.theories.cosmo import BoltzmannBase

from importlib import resources
import os
import numpy as np


class FixedLCDM(BoltzmannBase):
    """
    Theory class with fixed lensing and tensor BB spectra.

    The spectra can be scaled with the ``Al_scale`` and ``r`` parameters. The
    fixed spectra come from the inputs used to generate the BK simulations, and
    the lensed-LCDM spectrum should match the input to SPT-3G.

    """

    fixlcdm_fname: str | None = None
    lmax_theory: int = 600

    # BoltzmannBase expects this attribute during initialize(); FixedLCDM uses
    # explicit class options instead.
    extra_args: dict = {}

    r_parname = "r"
    Al_parname = "Al_scale"

    tens_bb: np.ndarray
    lens_bb: np.ndarray
    ells: np.ndarray
    invfac: np.ndarray

    def initialize(self):
        super().initialize()

        # have all objects, when relevant, start at ell=1 (bpwf, theory spec)

        if self.fixlcdm_fname is None:
            spec_path = resources.files("b1like").joinpath("fixlcdm_spec.npz")
        else:
            spec_path = self.fixlcdm_fname

        if not os.path.isfile(spec_path):
            raise FileNotFoundError(f"Fixed LCDM spectrum file not found: {spec_path}")

        spec = np.load(spec_path)  # tmp['cmbt'] gives r=1 bb spec; tmp['cmbl'] gives lensing bb spec
        tens_bb = spec['cmbt']
        lens_bb = spec['cmbl']
        tens_bb = np.concatenate([tens_bb, np.zeros(max(0, self.lmax_theory - len(tens_bb) + 1))])

        self.lmax_theory = int(self.lmax_theory)

        self.tens_bb = tens_bb[: self.lmax_theory + 1]
        self.lens_bb = lens_bb[: self.lmax_theory + 1]

        self.ells = np.arange(self.lmax_theory + 1)
        self.invfac = 2 * np.pi / (np.maximum(self.ells, 1) * (np.maximum(self.ells, 1) + 1))

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

    def calculate(self, state, want_derived=True, **params):
        """Return lens and tensor spectra."""
        dl_lens = self.lens_bb * state['params'][self.Al_parname]
        dl_tens = self.tens_bb * state['params'][self.r_parname]
        state["Dl_lensed_scalar"] = dict(ell=self.ells, bb=dl_lens)
        state["Cl_lensed_scalar"] = dict(ell=self.ells, bb=dl_lens * self.invfac)
        state["Dl_tensor"] = dict(ell=self.ells, bb=dl_tens)
        state["Cl_tensor"] = dict(ell=self.ells, bb=dl_tens * self.invfac)
        return

    def get_can_provide(self):
        return ["Cl_lensed_scalar", "Cl_tensor"]

    def get_Cl_lensed_scalar(self, ell_factor=False, **kwargs):
        """Return the lensed-scalar ``Cl`` values or ``Dl`` values."""
        if ell_factor:
            return self.current_state["Dl_lensed_scalar"].copy()
        else:
            return self.current_state["Cl_lensed_scalar"].copy()

    def get_Cl_tensor(self, ell_factor=False, **kwargs):
        """Return the tensor-only ``Cl`` values or ``Dl`` values."""
        if ell_factor:
            return self.current_state["Dl_tensor"].copy()
        else:
            return self.current_state["Cl_tensor"].copy()
