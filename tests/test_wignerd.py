import numpy as np
import pytest

from b1like.wignerd import get_product_spectra, get_product_spectra_camb


@pytest.fixture(scope="module")
def dust_product_spectra():
    lmax = 600
    ell = np.arange(lmax + 1)
    fac = ell * (ell + 1) / (2 * np.pi)
    inv_fac = np.zeros_like(fac, dtype=float)
    np.divide(1.0, fac, out=inv_fac, where=fac != 0)

    cl_dd = (np.maximum(ell, 1) / 80) ** -0.6 * inv_fac
    cl_bb = (np.maximum(ell, 1) / 80) ** -3
    cl_ee = 2 * cl_bb

    return lmax, np.array([cl_ee, cl_bb]), cl_dd


def test_product_spectra_consistency(dust_product_spectra):
    lmax, cl_pol, cl_dd = dust_product_spectra

    glq_result = get_product_spectra(cl_pol, cl_dd, lmax=lmax)
    wigner3j_result = get_product_spectra_camb(cl_pol, cl_dd, lmax=lmax)

    np.testing.assert_allclose(glq_result, wigner3j_result, rtol=1e-8, atol=1e-7)


@pytest.mark.benchonly
def test_glq(dust_product_spectra, benchmark):
    lmax, cl_pol, cl_dd = dust_product_spectra

    benchmark(get_product_spectra, cl_pol, cl_dd, lmax=lmax)


@pytest.mark.benchonly
def test_3j(dust_product_spectra, benchmark):
    lmax, cl_pol, cl_dd = dust_product_spectra

    benchmark(get_product_spectra_camb, cl_pol, cl_dd, lmax=lmax)
