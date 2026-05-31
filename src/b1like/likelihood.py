import warnings
import numpy as np
from cobaya.likelihoods.base_classes import CMBlikes

# from camb.mathutils import threej_coupling


class BKCompLike(CMBlikes):
    """
    Component-map version of Cobaya's ``CMBlikes`` likelihood.

    Map names must start with a standard component label, such as ``CMB_B``,
    ``Dust_B``, ``Sync_B``, or ``LT_B``. Each map component is assigned a
    pseudo-bandpass over theory components, and map-pair spectra are assembled
    from the theory components they share.

    Attributes
    ----------
    __COMPONENTS__: dict[str, dict[str, float]]
        Map-component pseudo-bandpasses.
    lpivot: int
        Foreground power-law pivot.
    """

    __COMPONENTS__ = {
        "CMB": {"lens": 1.0, "tensor": 1.0},
        "Dust": {"dust": 1.0},
        "Sync": {"sync": 1.0},
        "LT": {"lens": 1.0},
    }
    lpivot = 80

    @classmethod
    def parse_map_component(cls, map_name):
        """Return the standard component label encoded in a map name."""
        component = map_name.strip().split("_", maxsplit=1)[0]
        if component not in cls.__COMPONENTS__:
            accepted = ", ".join(cls.__COMPONENTS__)
            raise ValueError(f"Unknown map component {component!r}; expected one of {accepted}")
        return component

    def get_requirements(self):
        """Translate parent ``Cl`` requirements into component theory requirements."""
        req = super().get_requirements()
        cl_requirement = req.pop("Cl", None)

        if "lens" in getattr(self, "required_theory_components", ()):
            req["Cl_lensed_scalar"] = cl_requirement
        if "tensor" in getattr(self, "required_theory_components", ()):
            req["Cl_tensor"] = cl_requirement
        return req

    def init_params(self, ini):
        """Initialize parent state and component bookkeeping."""
        super().init_params(ini)

        self.log.debug("pcl_lmin: %s", self.pcl_lmin)
        self.log.debug("pcl_lmax: %s", self.pcl_lmax)

        self.map_components = [self.parse_map_component(mapname) for mapname in self.map_names]
        self.required_map_components = [self.map_components[i] for i in self.required_order]
        self.required_theory_components = {
            theory_component
            for map_component in self.required_map_components
            for theory_component in self.__COMPONENTS__[map_component]
        }
        for mapname, component in zip(self.map_names, self.map_components):
            self.log.debug("mapname: %s", mapname)

    def get_powerlaw_Dl(self, amplitude, alpha):
        """Return a foreground power-law spectrum in ``D_ell`` units."""
        ells = np.arange(self.pcl_lmax + 1)
        ells[0] = 1
        ratio = ells / self.lpivot
        return amplitude * ratio**alpha

    def get_lens_component_Dl(self, theory_cls, data_params, combination):
        return theory_cls["lens"].get(combination)

    def get_tensor_component_Dl(self, theory_cls, data_params, combination):
        return theory_cls["tensor"].get(combination)

    def get_dust_component_Dl(self, theory_cls, data_params, combination):
        if combination != "bb":
            return None
        return self.get_powerlaw_Dl(data_params["BBdust"], data_params["BBalphadust"])

    def get_sync_component_Dl(self, theory_cls, data_params, combination):
        if combination != "bb":
            return None
        return self.get_powerlaw_Dl(data_params["BBsync"], data_params["BBalphasync"])

    def get_theory_component_Dl(self, theory_component, theory_cls, data_params, combination):
        """Dispatch to the spectrum rule for one theory component."""
        try:
            get_Dl = getattr(self, f"get_{theory_component}_component_Dl")
        except AttributeError as exc:
            raise NotImplementedError(
                f"No theory spectrum rule implemented for component {theory_component!r}"
            ) from exc
        return get_Dl(theory_cls, data_params, combination)

    def get_component_pair_Dl(self, component_i, component_j, theory_cls, data_params, combination):
        """Return the weighted spectrum shared by two map components."""
        bandpass_i = self.__COMPONENTS__[component_i]
        bandpass_j = self.__COMPONENTS__[component_j]
        cls = None
        for theory_component in set(bandpass_i).intersection(bandpass_j):
            dl = self.get_theory_component_Dl(theory_component, theory_cls, data_params, combination)
            if dl is None:
                continue
            weighted_Dl = bandpass_i[theory_component] * bandpass_j[theory_component] * dl
            cls = weighted_Dl if cls is None else cls + weighted_Dl
        return cls

    def get_theory_map_cls(self, theory_cls, data_params=None):
        """Populate parent ``map_cls`` objects with component spectra."""
        for i in range(self.nmaps_required):
            ci = self.required_map_components[i]
            for j in range(i + 1):
                cj = self.required_map_components[j]
                CL = self.map_cls[i, j]
                combination = "".join([self.field_names[k] for k in CL.theory_ij]).lower()
                cls = self.get_component_pair_Dl(ci, cj, theory_cls, data_params, combination)

                if cls is not None:
                    tmp = cls[self.pcl_lmin : self.pcl_lmax + 1]
                    CL.CL[:] = tmp
                else:
                    CL.CL[:] = 0
        self.adapt_theory_for_maps(self.map_cls, data_params or {})

    def logp(self, **data_params):
        """Return the log likelihood for the current sampled parameters."""
        theory_cls = {}
        if "lens" in self.required_theory_components:
            theory_cls["lens"] = self.provider.get_Cl_lensed_scalar(ell_factor=True)
        if "tensor" in self.required_theory_components:
            theory_cls["tensor"] = self.provider.get_Cl_tensor(ell_factor=True)
        return self.log_likelihood(theory_cls, **data_params)

    def log_likelihood(self, dls, **data_params):
        """
        Return the log likelihood for precomputed component spectra.

        Parameters
        ----------
        dls: dict
            Theory spectra in ``D_ell`` units.
        **data_params
            Sampled or fixed likelihood parameters.

        Returns
        -------
        float
            Log-likelihood value.
        """
        return super().log_likelihood(dls, **data_params)
        self.get_theory_map_cls(dls, data_params)
        C = np.empty((self.nmaps, self.nmaps))
        big_x = np.empty(self.nbins_used * self.ncl_used)
        vecp = np.empty(self.ncl)
        chisq = 0
        if self.binned:
            binned_theory = self.get_binned_map_cls(self.map_cls)
        else:
            Cs = np.zeros((self.nbins_used, self.nmaps, self.nmaps))
            for i in range(self.nmaps):
                for j in range(i + 1):
                    CL = self.map_cls[i, j]
                    if CL is not None:
                        Cs[:, i, j] = CL.CL[self.bin_min - self.pcl_lmin : self.bin_max - self.pcl_lmin + 1]
                        Cs[:, j, i] = CL.CL[self.bin_min - self.pcl_lmin : self.bin_max - self.pcl_lmin + 1]
        for b in range(self.nbins_used):
            if self.binned:
                self.elements_to_matrix(binned_theory[b, :], C)
            else:
                C[:, :] = Cs[b, :, :]
            if self.cl_noise is not None:
                C += self.noise_matrix[b]
            if self.like_approx == 'exact':
                chisq += self.exact_chi_sq(C, self.bandpower_matrix[b], self.bin_min + b)
                continue
            elif self.like_approx == 'HL':
                try:
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        self.transform(C, self.bandpower_matrix[b], self.fiducial_sqrt_matrix[b])
                        if len(w) > 0:
                            if issubclass(w[-1].category, RuntimeWarning):
                                print("sqrt issue")
                                return -np.inf
                except np.linalg.LinAlgError:
                    self.log.debug("Likelihood computation failed.")
                    return -np.inf
            elif self.like_approx == 'gaussian':
                C -= self.bandpower_matrix[b]
            self.matrix_to_elements(C, vecp)
            big_x[b * self.ncl_used : (b + 1) * self.ncl_used] = vecp[self.cl_used_index]
        if self.like_approx == 'exact':
            return -0.5 * chisq
        return -0.5 * self._fast_chi_squared(self.covinv, big_x)
