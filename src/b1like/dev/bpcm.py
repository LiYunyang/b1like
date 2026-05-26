import numpy as np
from sampling import freq_scaling as fs
from sampling import model
from util import read_bandpass as rb
import mutils


def aps_getxind(naps, inda, indb):
    """Load lensed BB and tensor BB for scaling.

    Parameters
    ----------
    naps
        Number of APS entries.
    inda, indb
        Zero-indexed map locations.
    """
    nauto = -0.5 + np.sqrt(2 * naps + 0.25)

    assert inda <= indb
    assert inda < nauto
    assert indb < nauto

    if inda == indb:
        return int(inda)
    else:
        starti = np.sum([i for i in np.arange(nauto, nauto - (inda + 1), -1)])
        return int(starti + (indb - inda) - 1)


class BPCM:
    """
    Base class for constructing bandpower covariance matrices.

    Parameters
    ----------
    bandpowers
        Arrays containing SxS, SxN, NxS, and NxN terms with
        ``nmaps**2 x 4`` spectra.
    dictionary/mapping
        Mapping that tells the code where to find ``S_i x S_j``,
        ``S_i x N_j``, ``N_i x S_j``, and ``N_i x N_j`` between maps.

    Returns
    -------
    bpcm
        Bandpowers ordered as ``(1,1), (2,2), (3,3), ..., (1,2),
        (1,3), ..., (2,1), (2,3), ..., (3,1), (3,2)``.

    Notes
    -----
    Functions that could be common to all BPCM objects:
        raw_bpcm_terms()
        construct_snmask()
        construct_ell_mask()
        scale_bpcm()  #this should work, but need to be careful about indices
        aps_getxind()
        expt_field_order()
        order_expt_field()
    """

    def __init__(self, iscl=True):
        self.iscl = iscl

    def form_bp(self):
        """Input file specific array of SxS, SxN, NxS, and NxN spectra."""
        NotImplementedError('Call with experiment specific BPCM')
        return

    def initiate_bp_arr(self):
        """Return the standard dictionary for SxS, SxN, NxS, and NxN cross-spectra."""
        bp = {}
        bp['ss'] = np.zeros([self.nbins * self.n_spec, self.n_sim])
        bp['nn'] = np.zeros([self.nbins * self.n_spec, self.n_sim])
        bp['sn'] = np.zeros([self.nbins * self.n_spec, self.n_sim])
        bp['ns'] = np.zeros([self.nbins * self.n_spec, self.n_sim])

        return bp

    def raw_bpcm_terms(self):
        bp = self.form_bp()

        covmat = {}
        covmat['sig'] = np.cov(bp['ss'], rowvar=True, ddof=1)
        covmat['noi'] = np.cov(bp['nn'], rowvar=True, ddof=1)

        snns = np.concatenate((bp['sn'], bp['ns']), axis=0)
        snnscov = np.cov(snns, rowvar=True, ddof=1)
        N = bp['ss'].shape[0]
        covmat['sn1'] = snnscov[0:N, 0:N]  # s_i n_j s_k n_l
        covmat['sn2'] = snnscov[0:N, N : 2 * N]  # s_i n_j n_k s_l
        covmat['sn3'] = snnscov[N : 2 * N, 0:N]  # n_i s_j s_k n_l
        covmat['sn4'] = snnscov[N : 2 * N, N : 2 * N]  # n_i s_j n_k s_l

        return covmat

    def construct_snmask(self, ncbands=[], scbands=[]):  # noqa: C901
        """
        Construct signal-noise masks from ``mask_bpcm.m``.

        Signal-only terms keep all entries. Noise-only terms keep all entries
        for bands in ``ncbands`` and otherwise keep variance terms only.
        Signal-noise terms are kept when the noise contributions come from
        ``ncbands`` and the same experiment frequency only. Masks are computed
        for a single ``n_spec x n_spec`` block and repeated over all ell blocks.

        Parameters
        ----------
        ncbands
            Lists of experiment-frequency indices whose noise correlations
            should be kept. For example, ``[[0, 1], [5, 6]]`` keeps BK95/BK150
            and P44/P70 noise correlated within each pair.
        scbands
            Same structure as ``ncbands`` for signal correlations, when all
            signal components should not be assumed mutually correlated.
        """
        n_spec = self.n_spec

        noise_groups = np.arange(self.n_exptfreq, dtype=float)
        for ii in range(self.n_exptfreq):
            for jj, ncband in enumerate(ncbands):
                if ii in ncband:
                    noise_groups[ii] = jj + 0.5

        # for backwards compatibility where all signal are assumed correlated
        if scbands == []:
            scbands = [[np.arange(self.n_exptfreq)]]

        signal_groups = np.arange(self.n_exptfreq, dtype=float)
        for ii in range(self.n_exptfreq):
            for jj, scband in enumerate(scbands):
                if ii in scband:
                    signal_groups[ii] = jj + 0.5

        mask_sig = np.zeros([self.n_spec, self.n_spec])
        mask_noi = np.zeros([self.n_spec, self.n_spec])
        mask_sn1 = np.zeros([self.n_spec, self.n_spec])
        mask_sn2 = np.zeros([self.n_spec, self.n_spec])
        mask_sn3 = np.zeros([self.n_spec, self.n_spec])
        mask_sn4 = np.zeros([self.n_spec, self.n_spec])

        for i in range(n_spec):
            for j in range(n_spec):
                se1 = signal_groups[self.order[i]['e1']]
                se2 = signal_groups[self.order[i]['e2']]
                se3 = signal_groups[self.order[j]['e1']]
                se4 = signal_groups[self.order[j]['e2']]
                ne1 = noise_groups[self.order[i]['e1']]
                ne2 = noise_groups[self.order[i]['e2']]
                ne3 = noise_groups[self.order[j]['e1']]
                ne4 = noise_groups[self.order[j]['e2']]
                if ((se1 == se3) and (se2 == se4)) or ((se1 == se4) and (se2 == se3)):
                    mask_sig[i, j] = 1
                if ((ne1 == ne3) and (ne2 == ne4)) or ((ne1 == ne4) and (ne2 == ne3)):
                    mask_noi[i, j] = 1
                if (ne2 == ne4) and (se1 == se3):  # snsn
                    mask_sn1[i, j] = 1
                if (ne2 == ne3) and (se1 == se4):  # snns
                    mask_sn2[i, j] = 1
                if (ne1 == ne4) and (se2 == se3):  # nssn
                    mask_sn3[i, j] = 1
                if (ne1 == ne3) and (se2 == se4):  # nsns
                    mask_sn4[i, j] = 1

        mask = {}
        mask['sig'] = np.tile(mask_sig, [self.nbins, self.nbins])
        mask['noi'] = np.tile(mask_noi, [self.nbins, self.nbins])
        mask['sn1'] = np.tile(mask_sn1, [self.nbins, self.nbins])
        mask['sn2'] = np.tile(mask_sn2, [self.nbins, self.nbins])
        mask['sn3'] = np.tile(mask_sn3, [self.nbins, self.nbins])
        mask['sn4'] = np.tile(mask_sn4, [self.nbins, self.nbins])

        return mask

    def construct_ell_mask(self, loffdiag=1, bands=[], bands_offdiag=[]):
        """
        Construct the ell mask used by ``trim_bpcm.m``.

        The mask zeros BPCM elements that are more than the configured number
        of bins off the diagonal.

        Parameters
        ----------
        loffdiag
            Default number of off-diagonal bins to keep.
        bands
            Bands that have non-default off-diagonal bin widths.
        bands_offdiag
            Number of off-diagonal bins to keep for each band in ``bands``.
        """
        n_spec = self.n_spec
        nbins = self.nbins

        offdiagmask = np.ones([n_spec * nbins, n_spec * nbins])
        for ii in range(nbins):
            for jj in range(nbins):
                if np.abs(ii - jj) > loffdiag:
                    offdiagmask[ii * n_spec : (ii + 1) * n_spec, jj * n_spec : (jj + 1) * n_spec] = 0

        order = self.order
        for iband, band in enumerate(bands):
            bandmask = np.zeros([n_spec, n_spec])
            for ii in range(n_spec):
                for jj in range(n_spec):
                    bandlist = [order[ii]['e1'], order[ii]['e2'], order[jj]['e1'], order[jj]['e2']]
                    if band in bandlist:
                        bandmask[ii, jj] = 1
            bandmask_large = np.tile(bandmask, [nbins, nbins])
            for ii in range(nbins):
                for jj in range(nbins):
                    if np.abs(ii - jj) > bands_offdiag[iband]:
                        bandmask_large[ii * n_spec : (ii + 1) * n_spec, jj * n_spec : (jj + 1) * n_spec] = 0
            offdiagmask = np.maximum(offdiagmask, bandmask_large)

        return offdiagmask

    def scale_bpcm(self, no_scale=False):
        """
        Return the fully formed BPCM in unbiased bandpower-squared units.

        If sampling happens in biased bandpower or model space, this output
        has to be scaled to biased space.

        Parameters
        ----------
        no_scale
            Set to ``True`` when the raw BPCM model input matches ``C_fl`` and
            does not need explicit zeroing of terms, such as a single CMB
            component.
        """
        bp = self.form_bp()

        raw_bpcm = self.raw_bpcm_terms()

        snmask = self.construct_snmask(ncbands=self.ncbands, scbands=self.scbands)

        for key in snmask.keys():
            raw_bpcm[key] *= snmask[key]

        bpcm_out = np.zeros(raw_bpcm['sig'].shape)

        if no_scale:
            for key in raw_bpcm.keys():
                bpcm_out += raw_bpcm[key]
            return bpcm_out

        n_spec = self.n_spec
        nbins = self.nbins
        order = self.order
        n_field = np.size(self.fields)

        # signal_amp in BK scale_bpcm; model spec to be scaled to
        rtspec = self.model_spec_in_exptfreq(zcbands=self.zcbands)
        n_cpt = rtspec.shape[2]

        # bpwf gives unbiased bandpowers (bpwf factors needs to be normalized)
        bpwf = self.get_bpwf()

        f2n = self.field_arrayloc()

        # input unbiased sim spec
        C_fl = np.reshape(np.mean(bp['ss'], axis=1), (nbins, n_spec))
        # form vectors of length nbin*nspec

        findex_1 = [f2n[order[i]['f1']] + order[i]['e1'] * n_field for i in range(n_spec)] * nbins
        findex_2 = [f2n[order[i]['f2']] + order[i]['e2'] * n_field for i in range(n_spec)] * nbins
        lbin = np.concatenate([[i] * n_spec for i in range(nbins)])

        s1d = np.sqrt(C_fl[lbin, findex_1])[:, None]
        s2d = np.sqrt(C_fl[lbin, findex_2])[:, None]

        # here denom and numerator are both unbiased
        scale_1 = np.sqrt(np.sum(rtspec[:, findex_1, :] ** 2 * bpwf[:, :, None], axis=0)) / s1d
        scale_2 = np.sqrt(np.sum(rtspec[:, findex_2, :] ** 2 * bpwf[:, :, None], axis=0)) / s2d

        for ii in range(n_cpt):
            scale_matrix = np.outer((scale_1[:, ii] * scale_2[:, ii]), (scale_1[:, ii] * scale_2[:, ii]))
            bpcm_out = bpcm_out + scale_matrix * raw_bpcm['sig']

        for ii in range(n_cpt):
            for jj in range(ii + 1, n_cpt):
                scale_matrix = 0.5 * np.outer(
                    (scale_1[:, ii] * scale_2[:, jj]), (scale_1[:, ii] * scale_2[:, jj])
                )
                bpcm_out = bpcm_out + scale_matrix * raw_bpcm['sig']
                scale_matrix = 0.5 * np.outer(
                    (scale_1[:, jj] * scale_2[:, ii]), (scale_1[:, jj] * scale_2[:, ii])
                )
                bpcm_out = bpcm_out + scale_matrix * raw_bpcm['sig']
                scale_matrix = 0.5 * np.outer(
                    (scale_1[:, ii] * scale_2[:, jj]), (scale_1[:, jj] * scale_2[:, ii])
                )
                bpcm_out = bpcm_out + scale_matrix * raw_bpcm['sig']
                scale_matrix = 0.5 * np.outer(
                    (scale_1[:, jj] * scale_2[:, ii]), (scale_1[:, ii] * scale_2[:, jj])
                )
                bpcm_out = bpcm_out + scale_matrix * raw_bpcm['sig']

        for ii in range(n_cpt):
            scale_matrix = np.outer(scale_1[:, ii], scale_1[:, ii])
            bpcm_out = bpcm_out + scale_matrix * raw_bpcm['sn1']
            scale_matrix = np.outer(scale_1[:, ii], scale_2[:, ii])
            bpcm_out = bpcm_out + scale_matrix * raw_bpcm['sn2']
            scale_matrix = np.outer(scale_2[:, ii], scale_1[:, ii])
            bpcm_out = bpcm_out + scale_matrix * raw_bpcm['sn3']
            scale_matrix = np.outer(scale_2[:, ii], scale_2[:, ii])
            bpcm_out = bpcm_out + scale_matrix * raw_bpcm['sn4']

        bpcm_out = bpcm_out + raw_bpcm['noi']

        return bpcm_out

    def aps_getxind(self, naps, inda, indb):
        """
        Return the cross-spectrum index for two APS map indices.

        Parameters
        ----------
        naps
            Number of APS entries.
        inda, indb
            Zero-indexed map locations.
        """
        nauto = -0.5 + np.sqrt(2 * naps + 0.25)

        assert inda <= indb
        assert inda < nauto
        assert indb < nauto

        if inda == indb:
            return int(inda)
        else:
            starti = np.sum([i for i in np.arange(nauto, nauto - (inda + 1), -1)])
            return int(starti + (indb - inda) - 1)

    def expt_field_order(self):
        """
        Build the experiment-field ordering for bandpower arrays.

        Parameters
        ----------
        n_exptfreq
            Number of input experiment frequencies, such as B2, Keck 95, or
            Planck 217.
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
        n_field = np.size(self.fields)
        M = self.n_exptfreq * n_field

        exptloop = np.concatenate([[i] * n_field for i in np.arange(self.n_exptfreq)])
        fieldloop = [i for i in self.fields] * self.n_exptfreq

        order = {}
        ctr = 0

        for i in range(M):
            for j in np.arange(0, M - i):
                order[ctr] = {}
                order[ctr]['e1'] = exptloop[j]
                order[ctr]['f1'] = fieldloop[j]
                order[ctr]['e2'] = exptloop[j + i]
                order[ctr]['f2'] = fieldloop[j + i]
                # print("i=%i; j=%i"%(i,j))
                ctr += 1

        return order

    def order_expt_field(self):
        order = self.expt_field_order()

        revorder = {}
        for i in order.keys():
            revorder[(order[i]['e1'], order[i]['f1'], order[i]['e2'], order[i]['f2'])] = i

        return revorder

    def model_spec_in_exptfreq(self, zcbands=[]):
        """
        Compute the second part of ``model_rms.m``.

        Parameters
        ----------
        specin
            Dictionary of model spectra with ``'cmbt'``, ``'cmbl'``,
            ``'dust'``, and ``'sync'`` keys. Each component dictionary has
            ``TT``, ``EE``, and ``BB`` keys.
        zcbands
            Band names corresponding to entries in ``self.exptfreq`` where
            components are zeroed; for example, the LT band has no dust/sync
            signal, so ``scale_dust``, ``scale_sync``, and ``scale_tensor`` are
            set to 0.

        Returns
        -------
        model_rtspec
            Square root of the theory spectrum with shape
            ``[n_ell, n_exptfreq * n_field, n_cpt]``, where ``n_cpt`` is the
            number of independent components: dust, sync, CMB, tensor, and
            dust-sync correlation.
        """
        specin = self.model.model_spec(lmax=self.theory_lmax)

        n_field = np.size(self.fields)
        f2n = self.field_arrayloc()
        nell = len(specin['cmbl']['BB'])
        ncpt = 5

        rtspec = np.zeros([nell, self.n_exptfreq * n_field, ncpt])

        for ii, ef in enumerate(self.exptfreq):
            if ef in zcbands:
                scale_dust = scale_sync = scale_tensor = 0
            else:
                scale_dust = fs.freq_scaling(
                    self.bandpass[ef],
                    self.model.params['beta_d'],
                    self.model.params['T_d'],
                    self.model.params['dust_freq'],
                )
                scale_sync = fs.freq_scaling(
                    self.bandpass[ef], self.model.params['beta_s'], None, self.model.params['sync_freq']
                )
                scale_tensor = 1

            for f in self.fields:
                ind = ii * n_field + f2n[f]

                rtspec[:, ind, 0] = np.sqrt(specin['cmbl'][f + f])

                # scale sync and dust for this exptfreq, field
                sy = specin['sync'][f + f] * scale_sync**2
                du = specin['dust'][f + f] * scale_dust**2

                rtspec[:, ind, 1] = np.sqrt((1 - self.model.params['epsilon']) * sy)
                rtspec[:, ind, 2] = np.sqrt((1 - self.model.params['epsilon']) * du)
                rtspec[:, ind, 3] = np.sqrt(self.model.params['epsilon'] * (sy + du + 2 * np.sqrt(sy * du)))
                rtspec[:, ind, 4] = np.sqrt(specin['cmbt'][f + f] * scale_tensor)

        return rtspec

    def field_arrayloc(self):
        """Return the E/B field locations in BPCM-like arrays."""
        n_field = np.size(self.fields)
        f2n = {'B': 0}
        if n_field == 2:
            f2n = {'E': 0, 'B': 1}
        return f2n


class BPCM_B1(BPCM):
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


class BPCM_S4(BPCM):
    def __init__(
        self,
        cf,
        apsarr,
        finalbpwf,
        finaltf,
        bins,
        zcbands=[],
        params=None,
        iscl=True,
        n_sims=200,
        lmin=0,
        model=None,
    ):
        """Initialize the S4 BPCM object."""
        # config file
        # import dc08_config as cf

        self.exptfreq = cf.freqbeam.keys()  # freq of bands
        self.n_exptfreq = len(self.exptfreq)
        self.fields = ['B']
        self.bins = bins  # bins to use for analysis
        self.nbins = len(bins)
        self.order = self.expt_field_order()
        self.maporder = cf.maporder
        self.bandpass = cf.bandpass

        self.n_spec = len(self.order)
        self.apsarr = apsarr  # signoi aps arr
        self.finalbpwf = finalbpwf  # from final file (may not need?)
        self.finaltf = finaltf  # from final file (TF)
        self.zcbands = zcbands  # zc: bands to zero fg components
        self.n_sim = n_sims

        self.ncbands = []  # not involked for comp_sep paper
        self.scbands = []  # not involved for comp_sep paper

        # (freq,s/n) -> map idx (e.g. (20, 'n')->9
        # replace signoi_idx_per_exptband() in BPCM_BK
        self.revmaporder = dict([(value, key) for key, value in self.maporder.items()])

        if model is None:
            print("default S4 compsep paper mode")
            s4model = model.S4_model(config=cf, params=params, iscl=iscl, lmin=lmin)
            self.model = s4model
            self.theory_lmax = finalbpwf.shape[1] - 1  # bpwf starts at ell=0
        else:
            print("pipeb1 mode")
            self.model = model
            self.theory_lmax = finalbpwf.shape[1]  # bpwf starts at ell=1

        super().__init__(iscl=iscl)

    def form_bp(self, apply_rwf=True):
        """
        Form S4 bandpowers.

        Parameters
        ----------
        apply_rwf
            If ``True``, return unbiased bandpowers.

        Returns
        -------
        bp
            Dictionary with ``'ss'``, ``'sn'``, ``'ns'``, and ``'nn'`` keys.
            Each value has shape ``(nbins * nspec, n_sim)`` and is ordered by
            ``expt_field_order``. The input order is ``11, 22, 33, 44, 12, 13,
            14, 23, 24, 34``; the output order is ``11, 22, 33, 44, 12, 23,
            34, 13, 24, 14``.
        """
        apsarr = self.apsarr  # array format (naps, nbins, nsims)
        naps = np.shape(apsarr)[0]  # 171

        order = self.order

        nbins = self.nbins
        n_spec = self.n_spec
        n_sim = self.n_sim

        rwf = self.get_rwf()

        bp = self.initiate_bp_arr()

        for i in order.keys():
            thisrwf = rwf[:, i][:, None]

            for sn1 in ['sig', 'noi']:
                for sn2 in ['sig', 'noi']:
                    freqi = self.maporder[order[i]['e1']][0]
                    freqj = self.maporder[order[i]['e2']][0]
                    ii = self.revmaporder[(freqi, sn1[0])]
                    jj = self.revmaporder[(freqj, sn2[0])]
                    # print(freqi, freqj, ii, jj)
                    if ii <= jj:
                        loc = self.aps_getxind(naps, ii, jj)
                    else:
                        loc = self.aps_getxind(naps, jj, ii)

                    tmp = apsarr[loc, self.bins, :]
                    if apply_rwf:
                        if sn1[0] + sn2[0] == "ss":
                            tmp *= thisrwf
                    bp[sn1[0] + sn2[0]][i + np.arange(nbins) * n_spec, :] = tmp

        return bp

    def get_rwf(self):
        """
        Load the saved reciprocal transfer function from ``final.pkl``.

        Returns
        -------
        rwf
            ``nbin x n_spec`` array where ``biased bandpowers * rwf`` gives
            unbiased bandpowers.
        """
        rwf = np.zeros([self.nbins, self.n_spec])
        tf = self.finaltf  # finaltf in (n_spec, nbins) format
        order = self.order
        naps = tf.shape[0]

        for i in order.keys():
            xind = self.aps_getxind(naps, order[i]['e1'], order[i]['e2'])
            rwf[:, i] = 1.0 / tf[xind, self.bins]

        return rwf

    def get_bpwf(self):
        """
        Load and rearrange the saved bandpower window function from ``final.pkl``.

        Parameters
        ----------
        finalbpwf
            ``(all_nbins, nell)`` array, or ``(nbins * nspec, nell)`` array when
            each spectrum has a unique window function.

        Returns
        -------
        bpwf
            ``(nell, n_spec * nbins)`` array arranged for multiplication with
            theory spectra.
                [000000000000111....0000]
                ...
                [000000000000111....0000]
        """
        finalbpwf = self.finalbpwf
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


class BPCM_BK(BPCM):
    def __init__(
        self,
        exptfreq,
        fields,
        dataset,
        bins,
        bpdir,
        apsarr,
        finalbpwf,
        finalrwf,
        zcbands=['LT'],
        params=None,
        liketheorydir="/sdf/home/w/wlwu/bk/",
        iscl=False,
    ):
        """
        Initialize the BK BPCM object.

        Parameters
        ----------
        exptfreq
            Names of experiment frequencies in the order used by ``apsarr``.
        bpdir
            Dictionary keyed by ``B``, ``K``, ``L``, ``P``, and ``W`` that gives
            directories for reading bandpasses.
        """
        self.exptfreq = exptfreq
        self.n_exptfreq = len(exptfreq)
        self.fields = fields
        self.dataset = dataset
        self.bins = bins
        self.nbins = len(bins)
        self.order = self.expt_field_order()
        self.n_spec = len(self.order)
        self.apsarr = apsarr
        self.finalbpwf = finalbpwf
        self.finalrwf = finalrwf
        self.zcbands = zcbands
        self.n_sim = 499

        self.ncbands = [[0, 1, 14, 15]]  # not tested (should be behavior for bk14-sptpol
        self.scbands = []  # [] give all signal correlated (old behavior

        self.theory_lmax = finalbpwf['Cs_l'][0].shape[0]  # bpwf starts at ell=1

        bkmodel = model.BK_model(params=params, liketheorydir=liketheorydir)
        self.model = bkmodel

        self.bandpass = rb.read_bandpass(exptfreq, bpdir)

        super().__init__(iscl=iscl)

    def form_bp(self, apply_rwf=True):
        """
        Form BK bandpowers from the APS array.

        Parameters
        ----------
        apply_rwf
            If ``True``, return unbiased bandpowers.

        Returns
        -------
        bp
            Bandpowers used to form the raw covariance matrix. The input
            ``apsarr`` is the BK data product, typically from the ``566*``
            files or ``tmp['aps']`` after loading ``aps.mat``. Bandpowers in
            ``apsarr`` are biased and not suppression-factor corrected.
        """
        apsarr = self.apsarr
        naps = np.shape(apsarr)[0]

        order = self.order
        revorder = self.order_expt_field()
        mapdef = self.signoi_idx_per_exptband()

        nbins = self.nbins
        n_spec = self.n_spec
        n_sim = self.n_sim

        rwf = self.get_rwf()

        bp = self.initiate_bp_arr()

        for i in order.keys():
            # thisrwf = np.tile(rwf[:, i],[n_sim,1]).transpose()
            thisrwf = rwf[:, i][:, None]

            for sn1 in ['sig', 'noi']:
                for sn2 in ['sig', 'noi']:
                    xind_arr, bpind_arr = self.get_ind(
                        naps,
                        mapdef[order[i]['e1']][sn1],
                        mapdef[order[i]['e2']][sn2],
                        order[i]['f1'],
                        order[i]['f2'],
                    )
                    # import ipdb
                    # ipdb.set_trace()
                    for ii in range(len(xind_arr)):
                        tmp = apsarr[xind_arr[ii]][0][0][self.bins, bpind_arr[ii], :]
                        if apply_rwf:
                            tmp *= thisrwf
                        bp[sn1[0] + sn2[0]][i + np.arange(nbins) * n_spec, :] += tmp

        return bp

    def get_bpwf(self):
        """
        Extract the bandpower window function from the final file.

        This follows parts of ``like_read_final.m``. Loading ``final.mat`` gives
        ``finalbpwf = tmp['bpwf'][0]``. In ``finalbpwf[i][j]``, ``i`` runs over
        spectra, ``j=0`` gives ell, ``j=1`` gives ``(n_ell, nbin,
        field1xfield2)``, and ``j=2`` gives the sum of the window function
        along ell.

        Returns
        -------
        bpwf
            ``(n_ell, nbin * nspec)`` array, equivalent to
            ``likedata.bpwf.Cs_l``, ready to apply to theory ``Cl`` and output
            unbiased binned bandpowers.
        """
        finalbpwf = self.finalbpwf
        order = self.order
        nbins = self.nbins
        n_spec = self.n_spec
        nell = len(finalbpwf[0][0][0])
        naps = finalbpwf.shape[0]

        def bpwfind(f1, f2, sameband):
            spectra = ['TT', 'TE', 'EE', 'BB', 'TB', 'EB', 'ET', 'BT', 'BE']
            if sameband:
                # Use TE bpwf for TB, ET, BT; Use EE bpwf for EB, BE.
                ind = [0, 1, 2, 3, 1, 2, 1, 1, 2]
            else:
                # Use TE bpwf for TB; Use ET bpwf for BT; Use EE bpwf for EB, BE.
                ind = [0, 1, 2, 3, 1, 2, 6, 6, 2]
            return ind[spectra.index(f1 + f2)]

        bpwf = np.zeros([nell, nbins * n_spec])
        # bpcm order (bin 1; ...)
        for i in order.keys():
            xind = self.aps_getxind(naps, order[i]['e1'], order[i]['e2'])
            bwind = bpwfind(order[i]['f1'], order[i]['f2'], order[i]['e1'] == order[i]['e2'])
            bpwf[:, i + np.arange(nbins) * n_spec] = finalbpwf[xind][1][:, self.bins, bwind]

        return bpwf

    def get_rwf(self):
        """
        Extract the reciprocal window function from the final file.

        The output has ``expt_field_order`` and shape
        ``(nbins, n_spec * nfield)``, the same as ``likedata.bpwf.rwf``.

        Notes
        -----
        In BK definitions, ``unbiased bandpowers / rwf`` gives biased
        bandpowers, ``biased bandpowers * rwf`` gives unbiased bandpowers, and
        ``theory Cls * bpwf`` gives unbiased bandpowers for normalized ``bpwf``.
        """
        n_field = np.size(self.fields)
        rwf = np.zeros([self.nbins, self.n_spec * n_field])
        finalrwf = self.finalrwf
        order = self.order
        # naps used to get index in the finalrwf structure
        # naps and self.n_spec can be different because each aps in naps
        # can contribute EE, BB, TT, TE, etc. to the convariance matrix.
        naps = finalrwf.shape[0]

        for i in order.keys():
            xind = self.aps_getxind(naps, order[i]['e1'], order[i]['e2'])
            sameband = order[i]['e1'] == order[i]['e2']
            bpind = self.aps_getbpind(order[i]['f1'], order[i]['f2'], sameband)
            rwf[:, i] = finalrwf[xind][0][self.bins, bpind]

        return rwf

    def get_ind(self, naps, apsmap1s, apsmap2s, f1, f2):
        xind_arr = []
        bpind_arr = []

        for apsmap1 in apsmap1s:
            for apsmap2 in apsmap2s:
                if apsmap1 == apsmap2:
                    xind = self.aps_getxind(naps, apsmap1, apsmap2)
                    bpind = self.aps_getbpind(f1, f2, True)
                elif apsmap1 < apsmap2:
                    xind = self.aps_getxind(naps, apsmap1, apsmap2)
                    bpind = self.aps_getbpind(f1, f2, False)
                else:  # apsmap1 > apsmap2
                    xind = self.aps_getxind(naps, apsmap2, apsmap1)
                    bpind = self.aps_getbpind(f2, f1, False)

                xind_arr.append(xind)
                bpind_arr.append(bpind)

        return xind_arr, bpind_arr

    def aps_getbpind(self, field1, field2, sameband):
        """
        Return the APS bandpower index for two fields.

        Parameters
        ----------
        field1
            ``T``, ``E``, or ``B`` denoting the first input map to the spectrum.
        field2
            ``T``, ``E``, or ``B`` denoting the second input map to the
            spectrum.
        sameband
            ``True`` if ``field1`` and ``field2`` come from the same experiment
            frequency, for example ``B2_150``.
        """
        spectra = ['TT', 'TE', 'EE', 'BB', 'TB', 'EB', 'ET', 'BT', 'BE']
        # the ET, BT, BE is to provide T2E1, T2B1, E2B1 combos within the same aps
        # array (only exist for sameband=False)

        combo = field1 + field2

        if sameband:
            if combo in ['ET', 'BT', 'BE']:
                combo = field2 + field1  # flip

        ind = spectra.index(combo)

        if sameband:
            assert ind < 6

        return ind

    def signoi_idx_per_exptband(self):
        """
        Map experiment-frequency signal and noise components to APS maps.

        This is needed because the signal (LCDM) is shared among the 150 GHz BK
        map and the WMAP/Planck maps.
        """
        mapdef = {}
        if self.dataset == 'BK14_LT2894':
            n_exptfreq = self.n_exptfreq  # BK95,BK150,W023, P030, W033,
            # P044, P070, P100, P143, P217, P353, LT

            # APS array maps
            # Map 1 = lensed-LCDM, observed by Keck 95 GHz
            # Map 2 = lensed-LCDM, observed by Keck 150 GHz (also used for WMAP/Planck)
            # Maps 3--13 = noise sims for BK95, BK150, W023, P030, W033,
            #                             P044, P070, P100, P143, P217, P353
            # Map 14 = lensed-LCDM, observed by lensing template
            # Map 15 = noise sims for lensing template
            # Map 16 = additional noise terms for lensing template
            for i in range(1, n_exptfreq - 1):  # all except BK95,LT
                mapdef[i] = {}
                mapdef[i]['sig'] = [1]
                mapdef[i]['noi'] = [2 + i]
            mapdef[0] = {'sig': [0], 'noi': [2]}
            mapdef[11] = {'sig': [13], 'noi': [14, 15]}
        else:
            assert 0

        return mapdef
