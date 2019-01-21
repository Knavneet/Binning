def generate_ecdf_bins(data, col, show=False, quantiles=[0.25, 0.33, 0.5, 0.75]):
    from statsmodels.distributions.empirical_distribution import ECDF
    from scipy.interpolate import interp1d
    import random

    chk = ECDF(data[col])

    slope_changes = sorted(set(data[col]))

    sample_edf_values_at_slope_changes = [ chk(item) for item in slope_changes]
    invert_fn = interp1d(sample_edf_values_at_slope_changes, slope_changes)

    if show:
        points = np.linspace(0.1,1)
        y = invert_fn(points)

        plt.plot(points, y, 'r-', points, y, 'bo')
        plt.title('ECDF for Attribute:Amount')
        plt.ylabel('Cumulative Probability')
        plt.xlabel('Amount')
        plt.show()
    
    potential_bins = [invert_fn(pt) for pt in quantiles]
    return potential_bins
