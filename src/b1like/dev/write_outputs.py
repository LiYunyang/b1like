# collection of codes that output
# data products to files in formats that are
# expected by CosmoMC and Cobaya (currently)

import numpy as np
import pickle as pk


def reorder(bpcm, spec):
    """
    Reorder a ``(n_spec, nbins)`` array from APS order to BPCM order.

    The input order is ``11, 22, 33, 44, 12, 13, 14, 23, 24, 34``. The output
    order is ``11, 22, 33, 44, 12, 23, 34, 13, 24, 14``.
    """
    specout = np.zeros(spec.shape)
    naps = spec.shape[0]
    order = bpcm.order
    for i in order.keys():
        xind = bpcm.aps_getxind(naps, order[i]['e1'], order[i]['e2'])
        specout[i, :] = spec[xind, :]
    return specout


def dataset(cf, bpcm, fname, fname_simn=None, decorr="lin"):
    """
    Write the dataset definition file.

    The output contains the file names for ``Nl``, ``Clhat``, and related
    likelihood inputs.

    Parameters
    ----------
    cf
        Config file object.
    fname_simn
        Simulation file-name base. Defaults to ``fname`` when reading data.
    """
    if fname_simn is None:
        fname_simn = fname

    f = open(fname_simn + '.dataset', 'wt')
    f.write('like_approx = HL \n')

    mapnames = "".join(['{}_{} '.format(freq, field) for freq in bpcm.exptfreq for field in bpcm.fields])
    mapfield = "".join(['{} '.format(field) for i in bpcm.exptfreq for field in bpcm.fields])
    f.write('map_names = %s \n' % mapnames)
    f.write('map_fields = %s \n' % mapfield)

    f.write('binned = T\n')
    f.write('nbins = %i \n' % (len(bpcm.bins)))

    # cl lmax set to max_l in bpwf
    # assume bpwf with ell starts at 0
    bpwf = bpcm.get_bpwf()
    f.write('cl_lmin = 2\n')  # if 0; ell/ell_pivot fg gives errors
    f.write('cl_lmax = %i\n' % (bpwf.shape[0] - 1))

    bpstr = get_bpstr(cf, bpcm)

    f.write('cl_fiducial_file = %s_fiducial.dat\n' % fname)
    f.write('cl_fiducial_order = %s\n' % bpstr)
    f.write('cl_fiducial_includes_noise = F\n')
    f.write('cl_hat_file = %s_cl_hat.dat\n' % fname_simn)
    f.write('cl_hat_order = %s\n' % bpstr)
    f.write('cl_hat_includes_noise = F\n')
    f.write('cl_noise_file = %s_noise.dat\n' % fname)
    f.write('cl_noise_order = %s\n' % bpstr)
    f.write('covmat_fiducial = %s_covmat.dat\n' % fname)
    f.write('covmat_cl = %s\n' % bpstr)
    f.write('bin_window_files=windows/%s_bpwf_bin%s.txt\n' % (fname, '%u'))
    f.write('bin_window_in_order = %s\n' % bpstr)
    for freq in bpcm.exptfreq:
        for field in bpcm.fields:
            f.write('bandpass[{}_{}]=bandpass_{}.txt\n'.format(freq, field, freq))
    f.write('nuisance_params=%s.paramnames\n' % fname)
    f.write('fpivot_dust=353.0\n')
    f.write('fpivot_sync=23.0\n')
    f.write('fpivot_dust_decorr(1)=217.0\n')
    f.write('fpivot_dust_decorr(2)=353.0\n')
    f.write('fpivot_sync_decorr(1)=23.0\n')
    f.write('fpivot_sync_decorr(2)=33.0\n')
    f.write('lform_dust_decorr=%s\n' % decorr)
    f.write('lform_sync_decorr=%s\n' % decorr)
    f.close()

    return


def append_dataset(ilc_bands, fname, fname_simn=None):
    """Append input map bands for ILC bandpowers."""
    if fname_simn is None:
        fname_simn = fname
    print(fname_simn)

    f = open(fname_simn + '.dataset', 'a')

    for band in ilc_bands:
        f.write('bandpass[%i_B]=bandpass_%i.txt\n' % (band, band))

    f.close()

    return


def append_dataset_alt_model(fgres_form, fname_simn):
    print("deprecated")
    f = open(fname_simn + '.dataset', 'a')

    f.write('fgres_form=%s\n' % fgres_form)
    if fgres_form == "template":
        f.write('cl_dust_res=%s_cl_auxdust.dat\n' % fname_simn)
        f.write('cl_sync_res=%s_cl_auxsync.dat\n' % fname_simn)
    f.close()

    return


def Cfl(cf, bpcm, fname, scale_cfl=1.0):
    """
    Write fiducial model bandpowers for the H-L likelihood.

    The output text file contains ``bin_n`` followed by ``n_spec`` bandpowers.
    ``scale_cfl`` converts from ``K^2`` to ``uK^2``.

    Notes
    -----
    This duplicates code in ``scale_bpcm()``. The BPCM model must be the same
    as the ``Cfl`` model.
    """
    bpwf = bpcm.get_bpwf()
    # sqr root of theory spec (5 components)
    rtspec = bpcm.model_spec_in_exptfreq(bpcm.zcbands)
    n_cpt = rtspec.shape[2]

    f2n = bpcm.field_arrayloc()
    order = bpcm.order
    n_field = len(bpcm.fields)
    findex_1 = [f2n[order[i]['f1']] + order[i]['e1'] * n_field for i in range(bpcm.n_spec)] * bpcm.nbins
    findex_2 = [f2n[order[i]['f2']] + order[i]['e2'] * n_field for i in range(bpcm.n_spec)] * bpcm.nbins

    # inner sum along ell for bpwf
    # outer sum along components

    # (nbin* n_spec, n_cpt)
    aa = np.sqrt(np.sum(rtspec[:, findex_1, :] ** 2 * bpwf[:, :, None], axis=0))
    bb = np.sqrt(np.sum(rtspec[:, findex_2, :] ** 2 * bpwf[:, :, None], axis=0))

    # product of sq root spec from two exptfreq per component, sum all components
    Cfl = np.sum(aa * bb, axis=1).reshape(bpcm.nbins, bpcm.n_spec)
    Cfl *= scale_cfl

    bpstr = get_bpstr(cf, bpcm)

    f = open(fname + '_fiducial.dat', 'wt')
    f.write('#%s \n' % bpstr)
    for i in range(bpcm.nbins):
        # first column denotes n-th bin used in analysis
        tmp = "".join([' {0:0.6g}'.format(cl) for cl in Cfl[i, :]])
        f.write("%i %s\n" % (i + 1, tmp))
    f.close()

    return Cfl


def Cfl_b1(cf, bpcm, fname, nmaps):
    """Write B1 fiducial bandpowers for CMB, dust, optional sync, and LT."""
    bpwf = bpcm.get_bpwf()

    # modelspec = bpcm.model.model_spec(bpcm.theory_lmax)
    # lensBB = modelspec['cmbl']['BB']
    # dustBB = modelspec['dust']['BB']

    ell, sltt, slee, slbb, slte = np.loadtxt(bpcm.input_llcdm_fname, unpack=True)
    lensBB = np.concatenate([slbb[0:2], slbb])[: bpcm.theory_lmax + 1]

    ell = np.arange(bpcm.theory_lmax + 1)
    ell[0] = 1
    lpivot = 80
    dustBB = bpcm.params['A_d'] * (ell / lpivot) ** (bpcm.params['alpha_d'])

    # all in D_ell
    cmbxcmb = (bpwf.transpose() @ lensBB)[0 :: bpcm.n_spec]
    dusxdus = (bpwf.transpose() @ dustBB)[1 :: bpcm.n_spec]
    ltxlt = cmbxcmb
    cmbxdus = np.zeros_like(cmbxcmb)
    dusxlt = np.zeros_like(cmbxcmb)
    cmbxlt = cmbxcmb

    if nmaps == 3:
        Cfl = np.vstack([cmbxcmb, dusxdus, ltxlt, cmbxdus, dusxlt, cmbxlt]).transpose()
    elif nmaps == 4:
        syncBB = bpcm.params['A_s'] * (ell / lpivot) ** (bpcm.params['alpha_s'])
        synxsyn = (bpwf.transpose() @ syncBB)[2 :: bpcm.n_spec]
        dusxsyn = np.zeros_like(cmbxcmb)
        synxlt = np.zeros_like(cmbxcmb)
        cmbxsyn = np.zeros_like(cmbxcmb)

        Cfl = np.vstack(
            [cmbxcmb, dusxdus, synxsyn, ltxlt, cmbxdus, dusxsyn, synxlt, cmbxsyn, dusxlt, cmbxlt]
        ).transpose()

    bpstr = get_bpstr(cf, bpcm)

    f = open(fname + '_fiducial.dat', 'wt')
    f.write('#%s \n' % bpstr)
    for i in range(bpcm.nbins):
        # first column denotes n-th bin used in analysis
        tmp = "".join([' {0:0.6g}'.format(cl) for cl in Cfl[i, :]])
        f.write("%i %s\n" % (i + 1, tmp))
    f.close()

    return Cfl


def Clhat(cf, bpcm, fname, Clhat):
    """
    Write data or simulation bandpowers.

    Simulation bandpowers have shape ``(n_spec, all_nbins, nsim)``. ``Clhat``
    is passed as ``(n_spec, nbins)`` in the correct order:
    ``00, 11, ..., 01, 12, ..., 02, 13``.
    """
    f = open(fname + '_cl_hat.dat', 'wt')

    bpstr = get_bpstr(cf, bpcm)

    f.write('#%s \n' % bpstr)
    for i in range(bpcm.nbins):
        # first column denotes n-th bin used in analysis
        tmp = "".join([' {0:0.6g}'.format(cl) for cl in Clhat[:, i]])
        f.write("%i %s\n" % (i + 1, tmp))
    f.close()

    return


def Clfgres(cf, fname, clres):
    f = open(fname, 'wt')

    tmp = "".join(['{0:0.6g} '.format(cl) for cl in clres])
    f.write("%s\n" % (tmp))
    f.close()

    return


def Nl(cf, bpcm, fname, Nl):
    """Write input ``Nl`` with shape ``(nspec, nbins)``."""
    f = open(fname + '_noise.dat', 'wt')

    order = bpcm.order
    bpstr = get_bpstr(cf, bpcm)

    f.write('#%s \n' % bpstr)
    for i in range(bpcm.nbins):
        # first column denotes n-th bin used in analysis
        tmp = "".join([' {0:0.6g}'.format(nn) for nn in Nl[:, i]])
        f.write("%i %s\n" % (i + 1, tmp))
    f.close()

    return


def bpcm(cf, bpcm, fname, covmat):
    """Write the input covariance matrix with shape ``(n_spec * nbins, n_spec * nbins)``."""
    f = open(fname + '_covmat.dat', 'wt')

    bpstr = get_bpstr(cf, bpcm)

    f.write('#%s \n' % bpstr)
    for i in range(bpcm.nbins * bpcm.n_spec):
        tmp = "".join([' {0:0.6g}'.format(cv) for cv in covmat[i, :]])
        f.write("%s\n" % tmp)
    f.close()

    return


def get_bpstr(cf, bpcm):
    # convenience function
    order = bpcm.order
    mo = cf.maporder

    bpstr = "".join(
        [
            '{}_{}x{}_{} '.format(
                mo[order[s]['e1']][0], order[s]['f1'], mo[order[s]['e2']][0], order[s]['f2']
            )
            for s in order
        ]
    )
    return bpstr


def bandpass(bpcm):
    for freq in bpcm.exptfreq:
        f = open('bandpass_{}.txt'.format(freq), 'wt')
        thisbpass = bpcm.bandpass[freq]
        for i in range(thisbpass.shape[1]):
            f.write('%0.3f %0.6g \n' % (thisbpass[0][i], thisbpass[1][i]))
        f.close()

    return


def bpwf(bpcm, fname):
    bpwin = bpcm.get_bpwf()  # (nell, n_spec*nbin)
    n_spec = bpcm.n_spec

    for i in range(bpcm.nbins):
        f = open('windows/%s_bpwf_bin%i.txt' % (fname, i + 1), 'wt')
        for l in range(bpwin.shape[0]):  # ell, starts with 0
            tmp = "".join(' {0:0.6g}'.format(wf) for wf in bpwin[l, i * n_spec : (i + 1) * n_spec])
            f.write("%i %s\n" % (l, tmp))
        f.close()

    return


def params(fname):
    f = open(fname + '.paramnames', 'wt')
    f.write('BBdust         A_{B,\\mathrm{dust}}\n')
    f.write('BBsync         A_{B,\\mathrm{sync}}\n')
    f.write('BBalphadust    \\alpha_{B,\\mathrm{dust}}\n')
    f.write('BBbetadust     \\beta_{B,\\mathrm{dust}}\n')
    f.write('BBTdust        T_{\\mathrm{dust}}\n')
    f.write('BBalphasync    \\alpha_{B,\\mathrm{sync}}\n')
    f.write('BBbetasync     \\beta_{B,\\mathrm{sync}}\n')
    f.write('BBdustsynccorr \\epsilon_{\\mathrm{dust,sync}}\n')
    f.write('EEtoBB_dust     EE_{\\mathrm{dust}}/BB_{\\mathrm{dust}}\n')
    f.write('EEtoBB_sync     EE_{\\mathrm{sync}}/BB_{\\mathrm{sync}}\n')
    f.write('Delta_dust        \\Delta_{\\mathrm{dust}}\n')
    f.write('Delta_sync        \\Delta_{\\mathrm{sync}}\n')

    f.write('gamma_corr     \\gamma_{\\mathrm{corr}}\n')
    f.write('gamma_95       \\gamma_{\\mathrm{95}}\n')
    f.write('gamma_150      \\gamma_{\\mathrm{150}}\n')
    f.write('gamma_220      \\gamma_{\\mathrm{220}}\n')
    f.close()

    return
