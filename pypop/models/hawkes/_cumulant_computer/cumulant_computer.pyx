#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: language_level=3

import numpy as np  # For internal testing of the cython documentation
cimport numpy as np  # "cimport" is used to import special compile-time stuff

DTYPE_d = np.double
ctypedef np.double_t DTYPE_d_t


def _compute_covariance_ij(double mean_intensity_j, double integration_support,
                          np.ndarray[DTYPE_d_t, ndim=1] events_i,
                          np.ndarray[DTYPE_d_t, ndim=1] events_j,
                          double end_time):
    cdef:
        double val = 0.0
        double trend_C_j = 2 * integration_support * mean_intensity_j
        double max_time = end_time - integration_support
        int n_i = events_i.shape[0]
        int n_j = events_j.shape[0]
        double t_k_i = 0.0
        double t_l_j = 0.0
        int first_valid_l = 0
        int events_in_interval
        int k, l

    # Iterate over all events t_k_i
    for k in range(n_i):
        t_k_i = events_i[k]

        # Ignore events prior to `integration_support` to avoid edge effects in
        # the computation of the `events_in_interval`
        if t_k_i < integration_support:
            continue

        # Ignore events after this time to avoid edge effects in the
        # computation of the `events_in_interval`
        if t_k_i > max_time:
            break

        # Find largest t_l_j in the interval of interest
        # i.e., such that t_k_i - t_l_j < integration_support
        while first_valid_l < n_j:
            if t_k_i - events_j[first_valid_l] >= integration_support:
                first_valid_l += 1
            else:
                break

        # Start with the first valid event in range (index `first_valid_l`) and
        # iterate over events t_l_j until we get out of the range of interest
        l = first_valid_l
        events_in_interval = 0
        while l < n_j:
            t_l_j = events_j[l]
            abs_diff = abs(t_l_j - t_k_i)
            if abs_diff < integration_support:
                events_in_interval += 1
            else:
                break
            l += 1

        val += events_in_interval - trend_C_j

    val /= end_time

    return val


def compute_covariance(double integration_support,
                       list end_times,
                       np.ndarray[DTYPE_d_t, ndim=1] mean_intensity,
                       list multi_events):
    cdef:
        int i, j
        int nreal = len(multi_events)
        int dim = len(multi_events[0])
        DTYPE_d_t[:, :] C

    C = np.zeros((dim, dim))
    for r in range(nreal):
        for i in range(dim):
            for j in range(dim):
                c_ij = _compute_covariance_ij(
                    mean_intensity_j=mean_intensity[j],
                    integration_support=integration_support,
                    end_time=end_times[r],
                    events_i=multi_events[r][i],
                    events_j=multi_events[r][j])
                C[i, j] += c_ij / nreal
    return C


cdef _skewness_pos_part(double t_a_i, np.ndarray[DTYPE_d_t, ndim=1] events_j,
                        int first_valid_b, int n_j, double integration_support):
    # Compute number of events in range in dim j

    cdef int events_j_in_interval = 0

    # Find largest t_b_j in the interval of interest
    # i.e., such that t_a_i - t_b_j < integration_support
    while first_valid_b < n_j:
        if t_a_i - events_j[first_valid_b] >= integration_support:
            first_valid_b += 1
        else:
            break

    # Start with the first valid event in range (index `first_valid_b`) and
    # iterate over events t_b_j until we get out of the range of interest
    b = first_valid_b
    while b < n_j:
        t_b_j = events_j[b]
        abs_diff = abs(t_b_j - t_a_i)
        if abs_diff < integration_support:
            events_j_in_interval += 1
        else:
            break
        b += 1

    return first_valid_b, events_j_in_interval


cdef _skewness_neg_part(double mean_intensity_i,
                        double mean_intensity_k,
                        double end_time,
                        double integration_support,
                        np.ndarray[DTYPE_d_t, ndim=1] events_j,
                        np.ndarray[DTYPE_d_t, ndim=1] events_k,
                        int n_j,
                        int n_k):
    cdef:
        double width = 2 * integration_support
        double width_square = width * width
        int first_valid_c = 0
        double end_time_minus_width = end_time - width

        double val = 0.0
        double val_norm = 0.0
        double val_pos = 0.0
        double val_neg

        int n_j_minus_width
        int n_k_minus_width
        double abs_diff
        int b, c

    b = n_j - 1
    while events_j[b] > end_time_minus_width:
        b -= 1
    n_j_minus_width = b + 1

    c = n_k - 1
    while events_k[c] > end_time_minus_width:
        c -= 1
    n_k_minus_width = c + 1

    for b in range(n_j):
        t_b_j = events_j[b]

        while first_valid_c < n_k:
            if t_b_j - events_k[first_valid_c] >= width:
                first_valid_c += 1
            else:
                break

        c = first_valid_c
        while c < n_k:
            t_c_k = events_k[c]
            abs_diff = abs(t_b_j - t_c_k)
            if abs_diff < width:
                val_pos += width - abs_diff
            else:
                break
            c += 1

    val_neg = width_square * mean_intensity_k

    val_norm = mean_intensity_i / (end_time + 2 * integration_support)

    # print('val_norm =', val_norm, 'val_pos =', val_pos, 'val_neg =', val_neg)

    val = val_norm * (val_pos - val_neg)

    return val


def compute_skewness_ijk(double integration_support,
                         double end_time,
                         double mean_intensity_i,
                         double mean_intensity_j,
                         double mean_intensity_k,
                         np.ndarray[DTYPE_d_t, ndim=1] events_i,
                         np.ndarray[DTYPE_d_t, ndim=1] events_j,
                         np.ndarray[DTYPE_d_t, ndim=1] events_k):
    cdef:
        double val = 0.0

        double trend_j = 2 * integration_support * mean_intensity_j
        double trend_k = 2 * integration_support * mean_intensity_k
        double trend_ijk = trend_j * trend_k * mean_intensity_i

        double max_time = end_time - integration_support

        int n_i = events_i.shape[0]
        int n_j = events_j.shape[0]
        int n_k = events_k.shape[0]

        double t_a_i = 0.0
        double t_b_j = 0.0
        double t_c_k = 0.0
        int first_valid_b = 0
        int first_valid_c = 0
        int events_j_in_interval
        int events_k_in_interval
        int a, b, c

    # Iterate over all events t_a_i
    # print('Positive part...')
    for a in range(n_i):
        t_a_i = events_i[a]
        # print('a:', a, 't_a_i:', t_a_i)

        # Ignore events prior to `integration_support` to avoid edge effects in
        # the computation of the `events_in_interval`
        if t_a_i < integration_support:
            continue
        # Ignore events after this time to avoid edge effects in the
        # computation of the `events_in_interval`
        elif t_a_i > max_time:
            break
        else:
            # print('    -- ', 'ok')

            # Compute number of events in range in dim j
            first_valid_b, events_j_in_interval = _skewness_pos_part(
                t_a_i=t_a_i, events_j=events_j, first_valid_b=first_valid_b,
                n_j=n_j, integration_support=integration_support)
            # Compute number of events in range in dim k
            first_valid_c, events_k_in_interval = _skewness_pos_part(
                t_a_i=t_a_i, events_j=events_k, first_valid_b=first_valid_c,
                n_j=n_k, integration_support=integration_support)
            # Add positive part of cumulant
            val += (events_j_in_interval - trend_j) * (events_k_in_interval - trend_k)
    val /= end_time
    # print('val =', val)

    # print('Negartive part...')
    val -= _skewness_neg_part(mean_intensity_i=mean_intensity_i,
                              mean_intensity_k=mean_intensity_k,
                              integration_support=integration_support,
                              end_time=end_time,
                              events_j=events_j, events_k=events_k,
                              n_j=n_j, n_k=n_k)
    # print('val =', val)

    return val


def compute_skewness(double integration_support, list end_times,
                     np.ndarray[DTYPE_d_t, ndim=1] mean_intensity,
                     list multi_events):
    cdef:
        int r, i, j
        int nreal = len(multi_events)
        int dim = len(multi_events[0])
        DTYPE_d_t[:, :] Kc

    Kc = np.zeros((dim, dim))

    for r in range(nreal):
        for i in range(dim):
            for j in range(dim):
                kc_ij = compute_skewness_ijk(
                    integration_support=integration_support,
                    end_time=end_times[r],
                    mean_intensity_i=mean_intensity[i],
                    mean_intensity_j=mean_intensity[i],
                    mean_intensity_k=mean_intensity[j],
                    events_i=multi_events[r][i],
                    events_j=multi_events[r][i],
                    events_k=multi_events[r][j])
                Kc[i, j] += kc_ij / nreal
    return Kc


def compute_skewness_all(double integration_support, list end_times,
                         np.ndarray[DTYPE_d_t, ndim=1] mean_intensity,
                         list multi_events):
    cdef:
        int r, i, j, k
        int nreal = len(multi_events)
        int dim = len(multi_events[0])
        DTYPE_d_t d_ijk
        DTYPE_d_t[:, :, :] D

    D = np.zeros((dim, dim, dim))

    for r in range(nreal):
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    d_ijk = compute_skewness_ijk(
                        integration_support=integration_support,
                        end_time=end_times[r],
                        mean_intensity_i=mean_intensity[i],
                        mean_intensity_j=mean_intensity[j],
                        mean_intensity_k=mean_intensity[k],
                        events_i=multi_events[r][i],
                        events_j=multi_events[r][j],
                        events_k=multi_events[r][k])
                    D[i, j, k] += d_ijk / nreal
    return D


def compute_ground_truth_skewness(np.ndarray[DTYPE_d_t, ndim=1] L_vec,
                                  np.ndarray[DTYPE_d_t, ndim=2] C,
                                  np.ndarray[DTYPE_d_t, ndim=2] R):
    cdef:
        int i, j, k, m
        int dim = L_vec.shape[0]
        DTYPE_d_t[:, :, :] D

    D = np.zeros((dim, dim, dim))

    for i in range(dim):
        for j in range(i+1):
            for k in range(j+1):
                for m in range(dim):
                    D[i, j, k] += (R[i, m] * R[j, m] * C[k, m] +
                                   R[i, m] * C[j, m] * R[k, m] +
                                   C[i, m] * R[j, m] * R[k, m]
                                   - 2 * L_vec[m] * R[i, m] * R[j, m] * R[k, m])
    return D


def compute_covariance_ij_nphc(int r, int i, int j, double mean_intensity_j,
                          double integration_support, double end_time_r,
                          np.ndarray[DTYPE_d_t, ndim=1] events_i,
                          np.ndarray[DTYPE_d_t, ndim=1] events_j,
                          ):
    cdef:
        double res_C = 0.0
        double width = 2 * integration_support
        double trend_C_j = width * mean_intensity_j
        int n_i = events_i.shape[0]
        int n_j = events_j.shape[0]
        double t_k_i = 0.0
        double t_l_j = 0.0
        int last_l = 0
        int events_in_interval
        int k, l

    # print('i:', i, 'j:', j)
    # print('end_time:', end_time_r)
    # print('integration_support:', integration_support)
    # print('mean_intensity_j:', mean_intensity_j)
    # print('trend_C_j:', trend_C_j)

    for k in range(n_i):
        t_k_i = events_i[k]
        if t_k_i - integration_support < 0:
            continue

        # print('------', 'k:', k, 't_k_i:', t_k_i)

        while last_l < n_j:
            if events_j[last_l] <= t_k_i - width:
                last_l += 1
            else:
                break

        # print('    --', 'last_l', last_l)

        l = last_l
        events_in_interval = 0
        while l < n_j:
            t_l_j = events_j[l]
            # print('    --', 'l:', l, 't_l_j:', t_l_j)
            abs_diff = abs(t_l_j - t_k_i)
            if abs_diff < width:
                if abs_diff < integration_support:
                    events_in_interval += 1
                    # print('    --', '   yes +1 ->', events_in_interval)
                else:
                    # print('    --', '   no go next')
                    pass
            else:
                # print('    --', '   no break:', 'abs_diff =', abs_diff, '>', width, '= width')
                break
            l += 1

        if l == n_j:
            # print('    --', 'l == n_j  --> ignore')
            continue

        # print('    --', 'rec_C +=', events_in_interval, '-', trend_C_j)

        res_C += events_in_interval - trend_C_j

    res_C /= end_time_r

    return res_C
