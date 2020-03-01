'''Python port of MATLAB script.'''

import numpy as np

def nlgrappa_matlab(
        reduced_fourier_data, ORF, pe_loc, acs_data, acs_line_loc,
        num_block, num_column, times_comp):
    '''

    Parameters
    ----------
    reduced_fourier_data :
        undersampled k-space data
    ORF :
        outer reduction factor
    pe_loc :
        undersampled phase-encoding lines' location
    acs_data :
        auto-calibration signal data (middle region of k-space)
    acs_line_loc :
        auto-calibration signal lines' location
    num_block :
        number of blocks
    num_column :
        number of columns
    times_comp :
        times of the number of the first-order terms (the number of
        the second-order terms = time_comp X the number of the
        first-order terms)

    Returns
    -------
    full_fourier_data :
        reconstructed k-space (with ACS replacement)
    rec_img :
        reconstructed image
    coef0 :
        coefficients for reconstruction

    Notes
    -----
    time_comp parameter
    As the parameter time_comp increases, relevant second-order terms
    are added for reconstruction.
    When time_comp = 1

    Written by: Yuchou Chang, University of Wisconsin - Milwaukee
    Email: yuchou@uwm.edu; leiying@uwm.edu
    Created on Oct. 12, 2011

    References
    ----------
    .. [1] Y. Chang, D. Liang, L. Ying, "Nonlinear GRAPPA: A Kernal
           Approach to Parallel MRI Reconstruction". Magn. Reson.
           Med. 2012
    '''

    # Get dimensions and initialization
    d1_reduced, d2, num_coil = reduced_fourier_data.shape[:]
    d1 = d1_reduced*ORF

    # Not sure what this is about, but including from MATLAB script:
    if ORF == 3:
        d1 = d1_reduced*ORF - 2
    elif ORF == 5:
        d1 = d1_reduced*ORF - 4
    elif ORF == 6:
        d1 = d1_reduced*ORF - 2


    # Decide which lines are possible lines to fit
    all_acquired_line_loc = np.unique(np.sort(np.concatenate(
        (pe_loc, acs_line_loc))))
    combined_fourier_data = np.zeros(
        (d1, d2, num_coil), dtype=reduced_fourier_data.dtype)
    combined_fourier_data[pe_loc, ...] = reduced_fourier_data
    combined_fourier_data[acs_line_loc, ...] = acs_data
    ind_first = np.argwhere(all_acquired_line_loc == acs_line_loc[0])
    ind_last = np.argwhere(all_acquired_line_loc == acs_line_loc[-1])

    # Form the structure that indicates where lines are fitted
    valid_flag = False

    # Use a dictionary like a cell array:
    #line_group = cell(num_block, ORF-1)
    line_group = dict()
    for ii in range(num_block):
        for jj in range(ORF-1):
            line_group[(ii, jj)] = None

    for s in range(ind_first, ind_last):
        for mode in range(num_block):
            for offset in range(ORF-1):
                tmp = all_acquired_line_loc[s] - offset - (mode-1)*ORF
                tentative_line_ind = np.arange(
                    tmp, tmp + (num_block-1)*ORF, ORF)
                valid_flag = True
                for t in range(num_block):
                    if not np.argwhere(
                            all_acquired_line_loc == tentative_line_ind[t]):
                        valid_flag = False
                        break
                if valid_flag:
                    # TODO: make sure this does the right thing
                    line_group[(mode, offset)] = np.unique(
                        np.concatenate((
                            line_group[(mode, offset)],
                            all_acquired_line_loc[s],
                            tentative_line_ind)), axis=0)

    # Solve for the weighting coefficients
    fit_coef = np.zeros(
        (num_coil, ORF-1, num_block,
         (times_comp+1)*num_block*num_coil*num_column),
        dtype=reduced_fourier_data.dtype)

    for jj in range(num_coil):
        for offset in range(ORF-1):
            for mode in range(num_block):
                fit_mat = np.zeros((
                    num_block*num_coil*num_column,
                    d2*line_group[(mode, offset)].shape[0]))
                target_vec = np.zeros((
                    1, d2*line_group[(mode, offset)].shape[0]))

                for nn in range(line_group[(mode, offset)].shape[0]):
                    temp_data = combined_fourier_data[line_group[(mode, offset)][nn, 1:], ...]
                    temp_data = temp_data.transpose((0, 2, 1))
                    temp_data = np.reshape(temp_data, (num_block*num_coil, d2))
                    fit_mat[
                        (num_block*num_coil*np.floor(num_column/2)):(num_block*num_coil*np.ceil(num_column/2)),
                        ((nn-1)*d2):(nn*d2)] = temp_data
                    target_vec[((nn-1)*d2):(nn*d2)] = combined_fourier_data[line_group[(mode, offset)][nn, 0], :, jj]

                tmp = num_block*num_coil
                transfer_matrix = fit_mat[
                    (tmp*np.floor(num_column/2)):
                    tmp*(np.floor(num_column/2)+1), :]
                column_label = np.concatenate((
                    np.arange(np.floor(num_column/2))[::-1],
                    np.arange(np.floor(num_column/2))))
                for column_idx in range(num_column - 1):
                    if column_idx < np.floor(num_column/2):
                        tmp = num_block*num_coil
                        fit_mat[tmp*(column_idx-1):tmp*column_idx, :] = np.concatenate((
                            transfer_matrix[:, column_label[column_idx]:],
                            transfer_matrix[:, :column_label[column_idx]]))
                    else:
                        tmp = num_block*num_coil
                        fit_mat[tmp*column_idx:tmp*(column_idx+1), :] = np.concatenate((
                            transfer_matrix[:, -column_label[column_idx]:],
                            transfer_matrix[:, :-column_label[column_idx]]))

                fit_mat_dim = np.reshape(fit_mat, (
                    num_block*num_coil,
                    num_column,
                    d2*line_group[(mode, offset)].shape[0]))
                fit_mat_dim = np.reshape(fit_mat, (
                    num_block,
                    num_coil,
                    num_column,
                    d2*line_group[(mode, offset)].shape[0]))

                # nonlinear transformation
                new_fit_mat = np.zeros((
                    (times_comp+1)*fit_mat.shape[0],
                    fit_mat.shape[1]))
                tmp = num_block*num_coil*num_column
                new_fit_mat[:tmp, :] = fit_mat

                idx_comp = 0
                for idx_adj_1 in range(np.ceil(num_block/2)-1):
                    for idx_adj_2 in range(np.ceil(num_coil/2)-1):
                        for idx_adj_3 in range(np.ceil(num_column/2)-1):
                            fit_mat_shift = np.roll(
                                fit_mat_dim,
                                (idx_adj_1, idx_adj_2, idx_adj_3, 0))
                            fit_mat_shift = np.reshape(fit_mat_shift, (
                                num_block,
                                num_coil*num_column,
                                d2*line_group[(mode,offset)].shape[0]))
                            fit_mat_shift = np.reshape(fit_mat_shift, (
                                num_block*num_coil*num_column,
                                d2*line_group[(mode,offset)].shape[0]))

                            tmp = num_block*num_coil*num_column*(idx_comp+1)
                            new_fit_mat[tmp:tmp + num_block*num_coil*num_column, :] = fit_mat*fit_mat_shift
                            idx_comp += 1

                            if idx_comp >= times_comp:
                                break
                        if idx_comp >= times_comp:
                            break
                    if idx_comp >= times_comp:
                        break

                fit_coef[jj, offset, mode, :] = np.linalg.inv(
                    new_fit_mat.conj() @ new_fit_mat.T) @ new_fit_mat.conj() @ target_vec.T

    del temp_data

    # Generate the missing lines using superpositions
    candidate_fourier_data = np.zeros((d1, d2, num_coil, num_block))
    for mode in range(num_block):
        candidate_fourier_data[..., mode] = combined_fourier_data
    for ss in range(d1):
        if not np.argwhere(pe_loc == ss):
            offset = np.mod(ss, ORF)
            for mode in range(num_block):
                tmp = ORF*np.floor((ss-1)/ORF) - (mode-1)*ORF
                tentative_line_ind = np.arange(tmp, tmp + (num_block-1)*ORF, ORF)
                if np.max(tentative_line_ind) < d1 and np.min(tentative_line_ind) >= 0:
                    temp_data = combined_fourier_data[tentative_line_ind, ...]
                    temp_data = temp_data.transpose((0, 2, 1))
                    tmp = num_block*num_coil
                    fit_mat = np.zeros((tmp*num_column, d2))
                    temp_data = np.reshape(temp_data, (tmp, d2))

                    tmp = num_block*num_coil*np.floor(num_column/2)
                    fit_mat[tmp, :] = temp_data
                    column_label = np.concatenate((
                        np.arange(np.floor(num_column/2))[::-1],
                        np.arange(np.floor(num_column/2))))
                    for column_idx in range(num_column-1):
                        if column_idx < np.floor(num_column/2):
                            tmp = num_block*num_coil
                            fit_mat[tmp*(column_idx-1):tmp*column_idx, :] = np.concatenate((
                                temp_data[:, column_label[column_idx]:],
                                temp_data[:, :column_label[column_idx]]))
                        else:
                            tmp = num_block*num_coil
                            fit_mat[tmp*column_idx:tmp*(column_idx+1), :] = np.concatenate((
                                temp_data[:, -column_label[column_idx]:],
                                temp_data[:, :-column_label[column_idx]]))

                    fit_mat_dim = np.reshape(
                        fit_mat, (num_block*num_coil, num_column, d2))
                    fit_mat_dim = np.reshape(
                        fit_mat, (num_block, num_coil, num_column, d2))

                    # nonlinear transformation
                    new_fit_mat = np.zeros((
                        fit_mat.shape[0]*(times_comp+1), d2))
                    tmp = num_block*num_coil*num_column
                    new_fit_mat[:tmp, :] = fit_mat

                    idx_comp = 0
                    for idx_adj_1 in range(np.ceil(num_block/2)-1):
                        for idx_adj_2 in range(np.ceil(num_coil/2)-1):
                            for idx_adj_3 in range(np.ceil(num_column/2)-1):
                                fit_mat_shift = np.roll(
                                    fit_mat_dim,
                                    (idx_adj_1, idx_adj_2, idx_adj_3, 0))
                                fit_mat_shift = np.reshape(
                                    fit_mat_shift,
                                    (num_block, num_coil*num_column, d2))
                                fit_mat_shift = np.reshape(
                                    fit_mat_shift,
                                    (num_block*num_coil*num_column, d2))

                                tmp = num_block*num_coil*num_column*(idx_comp+1)
                                new_fit_mat[tmp:tmp+num_block*num_coil*num_column, :] = fit_mat*fit_mat_shift
                                idx_comp += 1

                                if idx_comp >= times_comp:
                                    break
                            if idx_comp >= times_comp:
                                break
                        if idx_comp >= times_comp:
                            break
                    for jj in range(num_coil):
                        candidate_fourier_data[ss, :, jj, mode] = (
                            np.squeeze(fit_coef[jj, offset, mode, :])).T @ new_fit_mat
                else:
                    candidate_fourier_data[ss, :, :, mode] = 0

    # Use ACS lines to obtain the goodness-of-fit coefficients
    gof_coef = np.zeros((num_coil, ORF-1, num_block))
    for jj in range(num_coil):
        for offset in range(ORF-1):
            fit_mat = []
            target_vec = []
            for ss in range(len(acs_line_loc)):
                if np.mod(acs_line_loc[ss], ORF) == offset:
                    valid_flag = True
                    for mode in range(num_block):
                        if not np.argwhere(line_group[(mode, offset)][:, 0] == acs_line_loc[ss]):
                            valid_flag = False
                            break
                    if valid_flag:
                        temp_mat = []
                        for mode in range(num_block):
                            temp_mat = np.concatenate((
                                temp_mat,
                                candidate_fourier_data[acs_line_loc[ss], :, jj, mode]), axis=1)
                        fit_mat = np.concatenate((fit_mat, temp_mat))
                        target_vec = np.concatenate((
                            target_vec,
                            combined_fourier_data[acs_line_loc[ss], :, jj]))
            gof_coef[jj, offset, :] = target_vec/fit_mat

    # Combine the data from different modes using goodness-of-fit
    full_fourier_data = combined_fourier_data
    for ss in range(d1):
        if not len(np.argwhere(all_acquired_line_loc == ss)):
            offset = np.mod(ss, ORF)
            for jj in range(num_coil):
                for mode in range(num_block):
                    full_fourier_data[ss, :, jj] += gof_coef[jj, offset, mode]*candidate_fourier_data[ss, :, jj, mode]

    full_fourier_data[acs_line_loc, ...] = acs_data

    # Image reconstruction using IFFT2 and sum-of-squares
    coil_img = np.fft.fftshift(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(np.fft.ifftshift(
        full_fourier_data, 0), 1)), 0), 1)
    rec_img = np.sqrt(np.sum(np.abs(coil_img)**2, axis=-1))

    coef0 = fit_coef

    return(full_fourier_data, rec_img, coef0)
