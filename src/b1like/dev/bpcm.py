from dataclasses import dataclass
from pathlib import Path
from functools import cached_property
import numpy as np

from ..likelihood import BKCompLike
from . import mutils

STANDARD_COMPONENT_NAMES = tuple(BKCompLike.__COMPONENTS__)


class BPCM_B1:
    def __init__(self, cf, apsarr, finalbpwf, finalrwf, bins, params=None, n_sims=200, scale_lt_noise=1.0):
        """
        Initialize the B1 BPCM object.

        Notes
        -----
        Differences from ``BPCM_S4``:

        * ``zcbands`` is not needed because bandpowers are scaled, not the BPCM.
        * ``iscl`` is not needed because the default is ``Dl``.
        * ``lmin`` is not needed because model input defaults to 0.

        Parameters
        ----------
        scale_lt_noise
            Factor used to multiply the LT noise spectra for the BPCM, matching
            different LT noise cases in the final file. This can be an array or
            scalar.
        """
        self.exptfreq = cf.freqbeam.keys()  # freq of bands
        self.n_exptfreq = len(self.exptfreq)
        self.fields = ['B']
        self.bins = bins  # bins to use for analysis
        self.nbins = len(bins)
        self.order = self.expt_field_order()
        self.maporder = cf.maporder
        self.bandpass = cf.bandpass

        self.n_spec = len(self.order)
        self.apsarr = apsarr  # signoi aps arr (the whole file tmp['aps'])
        self.bpwf = finalbpwf  # from final file (may not need?)
        self.n_sim = n_sims
        self.params = params

        nmaps = len(self.maporder)
        self.nmaps = nmaps
        self.ncbands = [list(np.arange(nmaps - 1))]  # CMB and dust (/sync) noise (correlated noise bands)
        self.scbands = [[0, int(nmaps - 1)]]  # CMB and LT signal (correlated signal)

        self.scale_lt_noise = scale_lt_noise

        self.bbidx = 3
        self.theory_lmax = finalbpwf.shape[1] - 1  # bpwf starts at ell=0
        self.input_llcdm_fname = (
            "/sdf/home/w/wlwu/repos/spt3g_software/simulations/data/camb/planck18_"
            "TTEEEE_lowl_lowE_lensing_highacc/planck2018_base_plikHM_TTTEEE_lowl_lowE_"
            "lensing_lensedCls.dat"
        )

        self.rwf = self.get_rwf(finalrwf, nmaps)  # from final file (tmp['supfac']['rwf'])
        super().__init__(iscl=False)

    def get_rwf(self, finalrwf, nmaps):
        mapping, rev_mapping, ori_pair2idx, tar_pair2idx, ori_idx2pair, tar_idx2pair = mutils.reorder_pairs(
            nmaps
        )
        rwf = {}
        for i in range(self.n_spec):
            rwf[i] = finalrwf[0][mapping[i]][self.bins, self.bbidx]

        return rwf

    def form_bp(self):
        # aps dictionary
        # indices to aps56 file aps56['aps'][xind['ss']][0][0] gives (nbins, TT/TE/EE/BB/..., nsims)
        # aps56['aps'][xind['ss']][0][1] gives ells
        # aps56['aps'][xind['ss']][0][2] gives 'cmb5xcmb5', etc.
        """Form the B1 bandpower dictionary."""

        def get_xind_simmean(nmaps):
            if nmaps == 3:  # CMB, Dust, LT
                if self.apsarr.shape[0] == 45:
                    # aps93566
                    xind = {
                        0: {'ss': 4, 'sn': 36, 'ns': 36, 'nn': 6},  # cmbxcmb
                        1: {'ss': 3, 'sn': 33, 'ns': 33, 'nn': 7},  # dustxdust   #use cmb as 's'
                        2: {'ss': 4, 'sn': 38, 'ns': 38, 'nn': 8},  # ltxlt
                        3: {'ss': 30, 'sn': 37, 'ns': 32, 'nn': 42},  # cmbxdust
                        4: {'ss': 30, 'sn': 34, 'ns': 37, 'nn': 44},  # dustxlt
                        5: {'ss': 4, 'sn': 38, 'ns': 36, 'nn': 43},  # cmbxlt
                    }
                elif self.apsarr.shape[0] == 15:
                    # aps166
                    # use CMB auto for LT auto for now; even though it has residual dust in it...
                    xind = {
                        0: {'ss': 0, 'sn': 6, 'ns': 6, 'nn': 2},
                        1: {'ss': 1, 'sn': 10, 'ns': 10, 'nn': 3},
                        2: {'ss': 0, 'sn': 8, 'ns': 8, 'nn': 4},
                        3: {'ss': 5, 'sn': 7, 'ns': 9, 'nn': 12},
                        4: {'ss': 5, 'sn': 11, 'ns': 7, 'nn': 14},
                        5: {'ss': 0, 'sn': 8, 'ns': 6, 'nn': 13},
                    }
                else:
                    assert 0

                simmeans = {
                    'cmbxcmb': self.apsarr[xind[0]['ss']][0][0][self.bins, self.bbidx, :].mean(1)
                    * self.rwf[0],
                    'dustxdust': self.apsarr[xind[1]['ss']][0][0][self.bins, self.bbidx, :].mean(1)
                    * self.rwf[1],  # FIXME change to type3
                    'ltxlt': self.apsarr[xind[2]['ss']][0][0][self.bins, self.bbidx, :].mean(1)
                    * self.rwf[0],  # use cmbxcmb for ltxlt signal
                }

            elif nmaps == 4:  # CMB, Dust, Sync, LT
                # 966 file
                xind = {
                    0: {'ss': 0, 'sn': 9, 'ns': 9, 'nn': 3},  # cmbxcmb
                    1: {'ss': 1, 'sn': 15, 'ns': 15, 'nn': 4},  # dustxdust
                    2: {'ss': 2, 'sn': 20, 'ns': 20, 'nn': 5},  # syncxsync
                    3: {'ss': 0, 'sn': 12, 'ns': 12, 'nn': 6},  # ltxlt
                    4: {'ss': 7, 'sn': 10, 'ns': 14, 'nn': 22},  # cmbxdust
                    5: {'ss': 13, 'sn': 16, 'ns': 19, 'nn': 25},  # dustxsync
                    6: {'ss': 18, 'sn': 21, 'ns': 11, 'nn': 27},  # syncxlt
                    7: {'ss': 8, 'sn': 11, 'ns': 18, 'nn': 23},  # cmbxsync
                    8: {'ss': 7, 'sn': 17, 'ns': 10, 'nn': 26},  # dustxlt
                    9: {'ss': 0, 'sn': 12, 'ns': 9, 'nn': 24},  # cmbxlt
                }
                simmeans = {
                    'cmbxcmb': self.apsarr[xind[0]['ss']][0][0][self.bins, self.bbidx, :].mean(1)
                    * self.rwf[0],
                    'dustxdust': self.apsarr[xind[1]['ss']][0][0][self.bins, self.bbidx, :].mean(1)
                    * self.rwf[1],
                    'syncxsync': self.apsarr[xind[2]['ss']][0][0][self.bins, self.bbidx, :].mean(1)
                    * self.rwf[2],
                    'ltxlt': self.apsarr[xind[3]['ss']][0][0][self.bins, self.bbidx, :].mean(1)
                    * self.rwf[0],  # apsarr points to cmbxcmb; need cmbxcmb rwf
                }

            return xind, simmeans

        xind, simmeans = get_xind_simmean(self.nmaps)

        scaling = self.scale_sigmap(simmeans)

        bp = self.initiate_bp_arr()

        rwf = self.rwf

        # scale LT noise
        rwf[2] *= self.scale_lt_noise

        def get_rwffac(nmaps, rwf):
            if nmaps == 3:  # CMB, Dust, LT
                rwffac = {
                    0: {'ss': rwf[0], 'sn': rwf[0], 'ns': rwf[0], 'nn': rwf[0]},
                    1: {'ss': rwf[1], 'sn': rwf[1], 'ns': rwf[1], 'nn': rwf[1]},
                    2: {
                        'ss': rwf[0],
                        'sn': np.sqrt(rwf[0] * rwf[2]),
                        'ns': np.sqrt(rwf[2] * rwf[0]),
                        'nn': rwf[2],
                    },
                    3: {
                        'ss': rwf[3],
                        'sn': np.sqrt(rwf[0] * rwf[1]),
                        'ns': np.sqrt(rwf[0] * rwf[1]),
                        'nn': np.sqrt(rwf[0] * rwf[1]),
                    },
                    4: {
                        'ss': np.sqrt(rwf[1] * rwf[0]),
                        'sn': np.sqrt(rwf[1] * rwf[2]),
                        'ns': np.sqrt(rwf[1] * rwf[0]),
                        'nn': np.sqrt(rwf[1] * rwf[2]),
                    },
                    5: {
                        'ss': rwf[0],
                        'sn': np.sqrt(rwf[0] * rwf[2]),
                        'ns': np.sqrt(rwf[0] * rwf[0]),
                        'nn': np.sqrt(rwf[0] * rwf[2]),
                    },
                }
            elif nmaps == 4:  # CMB, Dust, Sync, LT
                rwffac = {
                    0: {'ss': rwf[0], 'sn': rwf[0], 'ns': rwf[0], 'nn': rwf[0]},
                    1: {'ss': rwf[1], 'sn': rwf[1], 'ns': rwf[1], 'nn': rwf[1]},
                    2: {'ss': rwf[2], 'sn': rwf[2], 'ns': rwf[2], 'nn': rwf[2]},
                    3: {
                        'ss': rwf[0],
                        'sn': np.sqrt(rwf[0] * rwf[3]),
                        'ns': np.sqrt(rwf[3] * rwf[0]),
                        'nn': rwf[3],
                    },  # ltxlt
                    4: {
                        'ss': rwf[4],
                        'sn': np.sqrt(rwf[0] * rwf[1]),
                        'ns': np.sqrt(rwf[0] * rwf[1]),
                        'nn': np.sqrt(rwf[0] * rwf[1]),
                    },  # cmbxdust
                    5: {
                        'ss': rwf[5],
                        'sn': np.sqrt(rwf[1] * rwf[2]),
                        'ns': np.sqrt(rwf[1] * rwf[2]),
                        'nn': np.sqrt(rwf[1] * rwf[2]),
                    },  # dustxsync
                    6: {
                        'ss': np.sqrt(rwf[2] * rwf[0]),
                        'sn': np.sqrt(rwf[2] * rwf[3]),
                        'ns': np.sqrt(rwf[2] * rwf[0]),
                        'nn': np.sqrt(rwf[2] * rwf[3]),
                    },  # syncxlt
                    7: {
                        'ss': rwf[7],
                        'sn': np.sqrt(rwf[0] * rwf[2]),
                        'ns': np.sqrt(rwf[0] * rwf[2]),
                        'nn': np.sqrt(rwf[0] * rwf[2]),
                    },  # cmbxsync
                    8: {
                        'ss': np.sqrt(rwf[1] * rwf[0]),
                        'sn': np.sqrt(rwf[1] * rwf[3]),
                        'ns': np.sqrt(rwf[1] * rwf[0]),
                        'nn': np.sqrt(rwf[1] * rwf[3]),
                    },  # dustxlt
                    9: {
                        'ss': np.sqrt(rwf[0] * rwf[0]),
                        'sn': np.sqrt(rwf[0] * rwf[3]),
                        'ns': np.sqrt(rwf[0] * rwf[0]),
                        'nn': np.sqrt(rwf[0] * rwf[3]),
                    },  # cmbxlt
                }
            return rwffac

        rwffac = get_rwffac(self.nmaps, rwf)

        for i in xind.keys():
            for sn1 in ['s', 'n']:
                for sn2 in ['s', 'n']:
                    # print(i, sn1, sn2)
                    scale = scaling[i][sn1 + sn2][:, None] if not sn1 + sn2 == 'nn' else 1
                    rwf = rwffac[i][sn1 + sn2][:, None]
                    spec = self.apsarr[xind[i][sn1 + sn2]][0][0][self.bins, self.bbidx, :]
                    # print( spec.mean(1))
                    # print( (spec*scale*rwf).mean(1) )
                    bp[sn1 + sn2][i + np.arange(self.nbins) * self.n_spec, :] = spec * scale * rwf

        return bp

    def scale_sigmap(self, simmeans):
        """Return covariance-matrix scaling for a different fiducial model."""
        bpwf = self.get_bpwf()

        # SPT camb; in Dl, starts at ell=2
        ell, sltt, slee, slbb, slte = np.loadtxt(self.input_llcdm_fname, unpack=True)
        dlbb = np.concatenate([slbb[0:2], slbb])  # starts at ell=0 (to match bpwf)
        rtlens = np.sqrt((bpwf.transpose() @ dlbb[: self.theory_lmax + 1])[0 :: self.n_spec]) / np.sqrt(
            simmeans['cmbxcmb']
        )

        # Dust spec
        ell = np.arange(self.theory_lmax + 1)
        ell[0] = 1
        lpivot = 80
        dust = self.params['A_d'] * (ell / lpivot) ** (self.params['alpha_d'])
        rtdust = np.sqrt((bpwf.transpose() @ dust)[0 :: self.n_spec]) / np.sqrt(simmeans['dustxdust'])

        scaling = {}
        if self.nmaps == 3:
            scaling[0] = {'ss': rtlens * rtlens, 'sn': rtlens * 1, 'ns': 1 * rtlens, 'nn': 1}
            scaling[1] = {'ss': rtdust * rtdust, 'sn': rtdust * 1, 'ns': 1 * rtdust, 'nn': 1}
            scaling[2] = {'ss': rtlens * rtlens, 'sn': rtlens * 1, 'ns': 1 * rtlens, 'nn': 1}
            scaling[3] = {'ss': rtlens * rtdust, 'sn': rtlens * 1, 'ns': 1 * rtdust, 'nn': 1}
            scaling[4] = {'ss': rtlens * rtdust, 'sn': rtdust * 1, 'ns': 1 * rtlens, 'nn': 1}
            scaling[5] = {'ss': rtlens * rtlens, 'sn': rtlens * 1, 'ns': 1 * rtlens, 'nn': 1}
        elif self.nmaps == 4:
            sync = self.params['A_s'] * (ell / lpivot) ** (self.params['alpha_s'])
            rtsync = np.sqrt((bpwf.transpose() @ sync)[0 :: self.n_spec]) / np.sqrt(simmeans['syncxsync'])
            scaling[0] = {'ss': rtlens * rtlens, 'sn': rtlens * 1, 'ns': 1 * rtlens, 'nn': 1}
            scaling[1] = {'ss': rtdust * rtdust, 'sn': rtdust * 1, 'ns': 1 * rtdust, 'nn': 1}
            scaling[2] = {'ss': rtsync * rtsync, 'sn': rtsync * 1, 'ns': 1 * rtsync, 'nn': 1}
            scaling[3] = {'ss': rtlens * rtlens, 'sn': rtlens * 1, 'ns': 1 * rtlens, 'nn': 1}
            scaling[4] = {'ss': rtlens * rtdust, 'sn': rtlens * 1, 'ns': 1 * rtdust, 'nn': 1}  # cmbxdust
            scaling[5] = {'ss': rtdust * rtsync, 'sn': rtdust * 1, 'ns': 1 * rtsync, 'nn': 1}  # dustxsync
            scaling[6] = {'ss': rtsync * rtlens, 'sn': rtsync * 1, 'ns': 1 * rtlens, 'nn': 1}  # syncxlt
            scaling[7] = {'ss': rtlens * rtsync, 'sn': rtlens * 1, 'ns': 1 * rtsync, 'nn': 1}  # cmbxsync
            scaling[8] = {'ss': rtlens * rtdust, 'sn': rtdust * 1, 'ns': 1 * rtlens, 'nn': 1}  # dustxlt
            scaling[9] = {'ss': rtlens * rtlens, 'sn': rtlens * 1, 'ns': 1 * rtlens, 'nn': 1}  # cmbxlt

        return scaling

    def get_bpwf(self):
        """Return the bandpower window function as in ``BPCM_S4.get_bpwf``."""
        finalbpwf = self.bpwf
        # order  = self.order
        nbins = self.nbins
        n_spec = self.n_spec
        nell = len(finalbpwf[0])

        bpwf = np.zeros([nell, nbins * n_spec])

        if finalbpwf.shape[0] != nbins * n_spec:
            # same bpwf for all specs
            for ibin, iallbin in enumerate(self.bins):
                # loop to repeat
                for nn in np.arange(n_spec):
                    bpwf[:, ibin * n_spec + nn] = finalbpwf[iallbin, :]
        else:
            # unique bpwf for each spec
            for ibin, iallbin in enumerate(self.bins):
                bpwf[:, ibin * n_spec : (ibin + 1) * n_spec] = np.transpose(finalbpwf[iallbin::nbins, :])

        return bpwf


class BPCM:
    fields: list = ['B']

    def __init__(self, specs, maporder: list, bpwf: np.ndarray, ncbands, scbands, loffdiag, bin_slice: None):
        self.maporder = self.validate_maporder(maporder)
        self.specs = specs  # dict of arrays of shape (n_spec, nbins)
        assert np.ndim(bpwf) == 2

        if bin_slice is None:
            bin_slice = slice(None)
        self.rwf = self.bpwf2rwf(bpwf, bin_slice)
        self.bpwf = self.rwf @ bpwf
        self.nbins = self.bpwf.shape[0]
        bp = self.specs2bp(specs, self.rwf)
        self.ncbands = ncbands
        self.scbands = scbands
        self.cov = self.bp2cov(bp, ncbands=ncbands, scbands=scbands, loffdiag=loffdiag, raw=False)
        assert self.cov.shape[0] == self.nbins * self.n_spec, f"{self.cov.shape}, {self.nbins}, {self.n_spec}"

    @classmethod
    def from_file(cls, fname, maporder_mapping, bpwf, ncbands, scbands, bbidx=1, bin_slice=None, loffdiag=1):
        if bin_slice is None:
            bin_slice = slice(None)
        if isinstance(maporder_mapping, list):
            maporder_mapping = {_: _ for _ in maporder_mapping}
        maporder = list(maporder_mapping.keys())
        loaded = np.load(fname)
        keys = list(loaded['keys'])
        _odx = np.array([keys.index(v) for k, v in maporder_mapping.items()])

        _mdx = np.ix_(_odx, _odx)
        specs = {key: loaded[key][:, _mdx[0], _mdx[1], bbidx] for key in ['ss', 'nn', 'sn', 'ns', 'tot']}
        return cls(
            specs=specs,
            maporder=maporder,
            bpwf=bpwf,
            bin_slice=bin_slice,
            ncbands=ncbands,
            scbands=scbands,
            loffdiag=loffdiag,
        )

    def specs2bp(self, specs, rwf):
        """
        Convert the input spectra to bandpowers, applying the appropriate scaling.

        Parameters
        ----------
        specs: dict
            Input spectra, with keys 'ss', 'nn', 'sn', 'ns', 'tot'. Each value is an array of
            shape (nsims, n_comp, n_comp, _nbins).
        rwf: np.ndarray
            Inverse window function as a 2d array of shape (nbins, _nbins)

        Returns
        -------
        dict
            Bandpowers corresponding to the input spectra, with the same keys as `specs`. Each value is an
            array of shape (nbins*n_spec, nsims).
        """
        bp = dict()

        for key in ['ss', 'nn', 'sn', 'ns', 'tot']:
            nsims = specs[key].shape[0]
            bp[key] = np.zeros((self.n_spec * self.nbins, nsims))
            for k, o in self.order.items():
                _spec = np.einsum('ij,lj->li', rwf, specs[key][:, o['e1'], o['e2'], :])
                # assumed field is only B
                bp[key][k + np.arange(self.nbins) * self.n_spec, :] = _spec.T
        return bp

    def bp2cov(self, bp, ncbands, scbands, loffdiag=None, raw=False):
        snmask = self.construct_snmask(ncbands=ncbands, scbands=scbands, loffdiag=loffdiag)
        covmat = dict()
        covmat['sig'] = np.cov(bp['ss'], rowvar=True, ddof=1)
        covmat['noi'] = np.cov(bp['nn'], rowvar=True, ddof=1)

        snns = np.concatenate((bp['sn'], bp['ns']), axis=0)
        snnscov = np.cov(snns, rowvar=True, ddof=1)
        N = bp['ss'].shape[0]
        covmat['sn1'] = snnscov[0:N, 0:N]  # s_i n_j s_k n_l
        covmat['sn2'] = snnscov[0:N, N : 2 * N]  # s_i n_j n_k s_l
        covmat['sn3'] = snnscov[N : 2 * N, 0:N]  # n_i s_j s_k n_l
        covmat['sn4'] = snnscov[N : 2 * N, N : 2 * N]  # n_i s_j n_k s_l

        out = np.zeros_like(covmat['sig'])
        for key, _snmask in snmask.items():
            covmat[key] *= _snmask
            out += covmat[key]
        if raw:
            return covmat
        return out

    @property
    def n_fields(self):
        return len(self.fields)

    @property
    def n_comp(self):
        """Number of componnet maps."""
        return len(self.maporder)

    @property
    def n_spec(self):
        # total number of all cross/auto spectra
        n = self.n_comp * self.n_fields
        return n * (n + 1) // 2

    @cached_property
    def order(self):
        return self.get_component_field_order(n_comp=self.n_comp, fields=self.fields)

    @property
    def map_names(self):
        return [f'{comp}_{f}' for comp in self.maporder for f in self.fields]

    @property
    def map_fields(self) -> list:
        return [f for _ in self.maporder for f in self.fields]

    @cached_property
    def bp_str(self):
        mo = self.maporder
        bp_strs = [f"{mo[o['e1']]}_{o['f1']}x{mo[o['e2']]}_{o['f2']}" for o in self.order.values()]
        return " ".join(bp_strs)

    def get_cl_noise(self):
        out = np.mean(self.specs2bp(self.specs, self.rwf)['nn'], axis=-1).reshape(self.nbins, self.n_spec)
        groups = self.cbands2groups(n_comp=self.n_comp, cbands=self.ncbands)
        for j, o in self.order.items():
            if groups[o['e1']] != groups[o['e2']]:
                out[:, j] = 0
        return out

    def get_cl_fiducial(self):
        """Fiducial signal Cls, without adding noise."""
        out = np.mean(self.specs2bp(self.specs, self.rwf)['ss'], axis=-1).reshape(self.nbins, self.n_spec)
        groups = self.cbands2groups(n_comp=self.n_comp, cbands=self.scbands)
        for j, o in self.order.items():
            if groups[o['e1']] != groups[o['e2']]:
                out[:, j] = 0
        return out

    @staticmethod
    def bpwf2rwf(bpwf, bin_slice=None) -> np.ndarray:
        if bin_slice is None:
            bin_slice = slice(None)
        assert bpwf.ndim == 2, f"Expected 2d bpwf array, got shape {bpwf.shape}"
        # not applying bpwf debiasing
        # rwf = np.diag(np.ones(bpwf.shape[0]))[bin_slice]
        supfac = np.sum(bpwf, axis=1)
        rwf = np.zeros_like(supfac)
        rwf[supfac > 0] = 1 / supfac[supfac > 0]
        rwf = np.diag(rwf)[bin_slice]
        return rwf

    @staticmethod
    def validate_maporder(maporder: list):
        unknown = [component for component in maporder if component not in STANDARD_COMPONENT_NAMES]
        if unknown:
            accepted = ", ".join(STANDARD_COMPONENT_NAMES)
            raise ValueError(f"Unknown maporder component(s) {unknown}; expected one of {accepted}")
        return list(maporder)

    @staticmethod
    def get_component_field_order(n_comp: int, fields: list):
        """
        Build the experiment-field ordering for bandpower arrays.

        Parameters
        ----------
        n_comp
            Number of input componnets (experiment frequencies), such as CMB/Dust/Sync or B2_150/Planck 217.
        fields
            Field labels, for example ``['B']``.

        Returns
        -------
        dict
            Mapping from input-map pairs, such as ``B150_B x P217_E``, to their
            cross-spectrum location in the BPCM bandpower array. With one field,
            the order is autos, then 1-lag, 2-lag, and so on: ``11, 22, 33, 44,
            ..., 12, 23, 34, ..., 13, 24, ..., 14``.
        """
        n_field = np.size(fields)
        M = n_comp * n_field

        comp_loop = np.concatenate([[i] * n_field for i in np.arange(n_comp)])
        field_loop = [i for i in fields] * n_comp

        order = {}
        ctr = 0

        for i in range(M):
            for j in np.arange(0, M - i):
                order[ctr] = {}
                order[ctr]['e1'] = comp_loop[j]
                order[ctr]['f1'] = field_loop[j]
                order[ctr]['e2'] = comp_loop[j + i]
                order[ctr]['f2'] = field_loop[j + i]
                ctr += 1
        return order

    @staticmethod
    def cbands2groups(n_comp, cbands: list):
        groups = np.arange(n_comp, dtype=int)
        for jj, b in enumerate(cbands):
            for e in b:
                if e >= n_comp:
                    raise ValueError(f"Component index {e} in cbands is out of range for n_comp={n_comp}")

        for ii in range(n_comp):
            for jj, b in enumerate(cbands):
                if ii in b:
                    groups[ii] = -jj - 1
        return groups

    @classmethod
    def construct_snmask_block(cls, n_spec, n_comp, ncbands, scbands, order):
        """
        Construct signal-noise mask block from ``mask_bpcm.m``.

        Signal-only terms keep all entries. Noise-only terms keep all entries
        for bands in ``ncbands`` and otherwise keep variance terms only.
        Signal-noise terms are kept when the noise contributions come from
        ``ncbands`` and the same experiment frequency only. Masks are computed
        for a single ``n_spec x n_spec`` block and repeated over all ell blocks.

        Parameters
        ----------
        n_spec: int
            Total number of spectra (dimension of the output mask block).
        n_comp: int
            Number of component maps (experiment frequencies).
        ncbands: list of list
            Lists of component indices whose noise correlations should be kept.
            For example, ``[[0, 1, 2], [3]]`` keeps CMB/Dust/Sync noise correlated and LT noise uncorrelated.
        scbands: list of list
            Same structure as ``ncbands`` for signal correlations. `[[0, 3]]` would only keep signal
            correlations between CMB and LT.
        order:
            Mapping of the spectrum index to the corresponding component and field indices, as returned by
            ``get_component_field_order``.

        Returns
        -------
        dict of np.ndarray
            Dictionary of masks for each term type ('sig', 'noi', 'sn1', 'sn2', 'sn3', 'sn4') representing
            (`ssss`, `nnnn`, `snsn`, `snns`, `nssn`, `nsns`), where each mask is an ``n_spec x n_spec``
            array with entries of 1 for kept terms and 0 for zeroed terms.
        """
        noise_groups = cls.cbands2groups(n_comp, ncbands)
        signal_groups = cls.cbands2groups(n_comp, scbands)

        mask = {}
        for key in ['sig', 'noi', 'sn1', 'sn2', 'sn3', 'sn4']:
            mask[key] = np.zeros([n_spec, n_spec])

        for i in range(n_spec):
            oi = order[i]
            se1 = signal_groups[oi['e1']]
            se2 = signal_groups[oi['e2']]
            ne1 = noise_groups[oi['e1']]
            ne2 = noise_groups[oi['e2']]
            for j in range(n_spec):
                oj = order[j]
                se3 = signal_groups[oj['e1']]
                se4 = signal_groups[oj['e2']]
                ne3 = noise_groups[oj['e1']]
                ne4 = noise_groups[oj['e2']]

                if ((se1 == se3) and (se2 == se4)) or ((se1 == se4) and (se2 == se3)):
                    mask['sig'][i, j] = 1
                if ((ne1 == ne3) and (ne2 == ne4)) or ((ne1 == ne4) and (ne2 == ne3)):
                    mask['noi'][i, j] = 1
                if (ne2 == ne4) and (se1 == se3):  # snsn
                    mask['sn1'][i, j] = 1
                if (ne2 == ne3) and (se1 == se4):  # snns
                    mask['sn2'][i, j] = 1
                if (ne1 == ne4) and (se2 == se3):  # nssn
                    mask['sn3'][i, j] = 1
                if (ne1 == ne3) and (se2 == se4):  # nsns
                    mask['sn4'][i, j] = 1
        return mask

    def construct_snmask(self, ncbands=None, scbands=None, loffdiag=None):
        mask = self.construct_snmask_block(
            n_spec=self.n_spec, n_comp=self.n_comp, ncbands=ncbands, scbands=scbands, order=self.order
        )
        n_off = self.nbins if loffdiag is None else loffdiag
        for key in ['sig', 'noi', 'sn1', 'sn2', 'sn3', 'sn4']:
            kernel = np.triu(np.tril(np.ones(self.nbins), k=n_off), k=-n_off)
            mask[key] = np.kron(kernel, mask[key])
        return mask


@dataclass
class CobayaWriter:
    """
    Write the final files consumed by Cobaya's ``CMBlikes`` reader.

    The writer is intentionally thin: a B1 BPCM object owns ordering,
    covariance, and window construction, while this class normalizes those
    arrays into the text files named by a Cobaya ``.dataset`` manifest.

    Array conventions are fixed:
    ``cl_hat``, ``cl_noise``, and ``cl_fiducial`` are ``(nbins, nspec)``;
    ``covmat`` is ``(nbins*nspec, nbins*nspec)``; and ``bpwf`` is
    ``(nbins, nell)``.
    """

    bpcm: BPCM
    outdir: str | Path
    cl_noise: np.ndarray | None = None
    cl_fiducial: np.ndarray | None = None

    like_approx: str = "HL"
    cl_lmin: int = 2
    cl_lmax: int | None = None
    cl_hat_includes_noise: bool = True
    cl_fiducial_includes_noise: bool = False
    include_nuisance_params: bool = False
    nuisance_params: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        self.outdir = Path(self.outdir)
        self.n_spec = self.bpcm.n_spec
        self.map_names = self.bpcm.map_names
        self.map_fields = self.bpcm.map_fields
        self.bp_str = self.bpcm.bp_str
        self.bpwf = np.asarray(self.bpcm.bpwf)

        self.cl_noise = np.asarray(self.cl_noise) if self.cl_noise is not None else self.bpcm.get_cl_noise()
        self.cl_fiducial = (
            np.asarray(self.cl_fiducial) if self.cl_fiducial is not None else self.bpcm.get_cl_fiducial()
        )

        if self.cl_lmax is None:
            self.cl_lmax = self.bpwf.shape[1] - 1
        else:
            self.bpwf = self.bpwf[:, : self.cl_lmax + 1]

    @property
    def nbins(self):
        return int(self.bpcm.nbins)

    @property
    def covmat(self):
        return np.asarray(self.bpcm.cov)

    def validate(self):
        expected = (self.nbins, self.n_spec)
        if len(self.map_names) != len(self.map_fields):
            raise ValueError("map_names and map_fields must have the same length")

        for name, arr in (("cl_noise", self.cl_noise), ("cl_fiducial", self.cl_fiducial)):
            if arr.shape != expected:
                raise ValueError(f"{name} must have shape {expected}; got {arr.shape}")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} contains non-finite values")

        cov_shape = (self.nbins * self.n_spec, self.nbins * self.n_spec)
        if self.covmat.shape != cov_shape:
            raise ValueError(f"covmat must have shape {cov_shape}; got {self.covmat.shape}")
        if not np.all(np.isfinite(self.covmat)):
            raise ValueError("covmat contains non-finite values")
        if not np.allclose(self.covmat, self.covmat.T):
            raise ValueError("covmat must be symmetric")

        assert self.bpwf.shape == (self.nbins, self.cl_lmax + 1)

    def write(self, stem):
        """Write shared Cobaya ``CMBlikes`` products."""
        self.validate()
        self.outdir.mkdir(parents=True, exist_ok=True)
        windows_dir = self.outdir / "windows"
        windows_dir.mkdir(exist_ok=True)

        self._write_cl_file(self.outdir / f"{stem}_noise.dat", self.cl_noise)
        self._write_cl_file(self.outdir / f"{stem}_fiducial.dat", self.cl_fiducial)
        self._write_covmat(self.outdir / f"{stem}_covmat.dat")
        self._write_windows(windows_dir, stem)
        self._write_paramnames(self.outdir / f"{stem}.paramnames")

    def write_data(self, cl_hat, stem, *, data_stem=None):
        assert cl_hat.shape == (self.nbins, self.n_spec), (
            f"cl_hat must have shape {(self.nbins, self.n_spec)}; got {cl_hat.shape}"
        )
        self.outdir.mkdir(parents=True, exist_ok=True)
        data_stem = data_stem or stem
        dataset_file = self.outdir / f"{data_stem}.dataset"
        cl_hat_file = self.outdir / f"{data_stem}_cl_hat.dat"
        self._write_dataset(dataset_file, stem, data_stem)
        self._write_cl_file(cl_hat_file, cl_hat)
        return str(dataset_file), str(cl_hat_file)

    def _write_dataset(self, path, stem, data_stem):
        with path.open("wt", encoding="utf-8") as f:
            f.write(f"like_approx = {self.like_approx}\n")
            f.write(f"map_names = {' '.join(self.map_names)}\n")
            f.write(f"map_fields = {' '.join(self.map_fields)}\n")
            f.write("binned = T\n")
            f.write(f"nbins = {self.nbins}\n")
            f.write(f"cl_lmin = {self.cl_lmin}\n")
            f.write(f"cl_lmax = {self.cl_lmax}\n")
            f.write(f"cl_fiducial_file = {stem}_fiducial.dat\n")
            f.write(f"cl_fiducial_order = {self.bp_str}\n")
            f.write(f"cl_fiducial_includes_noise = {'T' if self.cl_fiducial_includes_noise else 'F'}\n")
            f.write(f"cl_hat_file = {data_stem}_cl_hat.dat\n")
            f.write(f"cl_hat_order = {self.bp_str}\n")
            f.write(f"cl_hat_includes_noise = {'T' if self.cl_hat_includes_noise else 'F'}\n")
            f.write(f"cl_noise_file = {stem}_noise.dat\n")
            f.write(f"cl_noise_order = {self.bp_str}\n")
            f.write(f"covmat_fiducial = {stem}_covmat.dat\n")
            f.write(f"covmat_cl = {self.bp_str}\n")
            f.write(f"bin_window_files = windows/{stem}_bpwf_bin%u.txt\n")
            f.write(f"bin_window_in_order = {self.bp_str}\n")
            if self.include_nuisance_params:
                f.write(f"nuisance_params = {stem}.paramnames\n")

    def _write_cl_file(self, path, values):
        with path.open("wt", encoding="utf-8") as f:
            f.write(f"# {self.bp_str}\n")
            for i, row in enumerate(values, start=1):
                f.write(f"{i} " + " ".join(f"{x:0.6g}" for x in row) + "\n")

    def _write_covmat(self, path):
        with path.open("wt", encoding="utf-8") as f:
            f.write(f"# {self.bp_str}\n")
            for row in self.covmat:
                f.write(" ".join(f"{x:0.6g}" for x in row) + "\n")

    def _write_windows(self, windows_dir, stem):
        for b in range(self.nbins):
            path = windows_dir / f"{stem}_bpwf_bin{b + 1}.txt"
            with path.open("wt", encoding="utf-8") as f:
                for ell, weight in enumerate(self.bpwf[b]):
                    row = np.full(self.n_spec, weight)
                    f.write(f"{ell} " + " ".join(f"{x:0.6g}" for x in row) + "\n")

    def _write_paramnames(self, path):
        if not self.include_nuisance_params:
            return
        with path.open("wt", encoding="utf-8") as f:
            for name, label in self.nuisance_params:
                f.write(f"{name:<18} {label}\n")
