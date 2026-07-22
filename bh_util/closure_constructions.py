import ehtim as eh
import numpy as np
from scipy.sparse import block_diag


def get_minimal_cphases(obs):
    """
    Given an eht-imaging obs file this will construct the set of minimal closure phases that are non-degenerate
    and maximize the SNR.

    The function returns

    - The array of closure phases measured from the data with time, stations, and uv coordinates
    - The sparse design matrix that maps from the set of visibilties in obs (i.e., obs.data["vis"]) to the closure phase array returned
    - The pairs of the uv-coordinates for each visibility


    This method is adapted from code written by Dom Pesce (see also Blackburn et al, 2021).


    """
    print("Working on getting minimal cphases...")
    obs.reorder_tarr_snr()
    obs.add_cphase(count="max")

    # organize some info
    time_vis = obs.data["time"]
    time_cp = obs.cphase["time"]
    t1_vis = obs.data["t1"]
    t2_vis = obs.data["t2"]

    # Determine the number of timestamps containing a closure triangle
    timestamps_cp = np.unique(time_cp)
    N_times_cp = len(timestamps_cp)

    # loop over all timestamps
    obs_cphase_arr = []
    uvecs = []
    vvecs = []
    visvecs = []
    sigmavecs = []
    design_mats = []
    for i in np.arange(0, N_times_cp, 1):
        # get the current timestamp
        ind_here_cp = time_cp == timestamps_cp[i]
        time_here = time_cp[ind_here_cp]

        # copy the cphase table for this timestamp
        obs_cphase_orig = np.copy(obs.cphase[ind_here_cp])

        # sort by cphase SNR
        snr = 1.0 / ((np.pi / 180.0) * obs_cphase_orig["sigmacp"])
        ind_snr = np.argsort(snr)
        obs_cphase_orig = obs_cphase_orig[ind_snr]
        snr = snr[ind_snr]

        # organize the closure phase stations
        cp_ant1_vec = obs_cphase_orig["t1"]
        cp_ant2_vec = obs_cphase_orig["t2"]
        cp_ant3_vec = obs_cphase_orig["t3"]

        # get the number of time-matched baselines
        ind_here_bl = time_vis == timestamps_cp[i]
        B_here = ind_here_bl.sum()

        # organize the time-matched baseline stations
        bl_ant1_vec = t1_vis[ind_here_bl]
        bl_ant2_vec = t2_vis[ind_here_bl]

        # initialize the design matrix
        design_mat = np.zeros((ind_here_cp.sum(), B_here))

        # fill in each row of the design matrix
        for ii in range(ind_here_cp.sum()):
            # determine which stations are in this triangle
            ant1_here = cp_ant1_vec[ii]
            ant2_here = cp_ant2_vec[ii]
            ant3_here = cp_ant3_vec[ii]

            # matrix entry for first leg of triangle
            ind1_here = (bl_ant1_vec == ant1_here) & (bl_ant2_vec == ant2_here)
            if ind1_here.sum() == 0.0:
                ind1_here = (bl_ant1_vec == ant2_here) & (bl_ant2_vec == ant1_here)
                val1_here = -1.0
            else:
                val1_here = 1.0
            design_mat[ii, ind1_here] = val1_here

            # matrix entry for second leg of triangle
            ind2_here = (bl_ant1_vec == ant2_here) & (bl_ant2_vec == ant3_here)
            if ind2_here.sum() == 0.0:
                ind2_here = (bl_ant1_vec == ant3_here) & (bl_ant2_vec == ant2_here)
                val2_here = -1.0
            else:
                val2_here = 1.0
            design_mat[ii, ind2_here] = val2_here

            # matrix entry for third leg of triangle
            ind3_here = (bl_ant1_vec == ant3_here) & (bl_ant2_vec == ant1_here)
            if ind3_here.sum() == 0.0:
                ind3_here = (bl_ant1_vec == ant1_here) & (bl_ant2_vec == ant3_here)
                val3_here = -1.0
            else:
                val3_here = 1.0
            design_mat[ii, ind3_here] = val3_here

        # determine the expected size of the minimal set
        N_min = np.linalg.matrix_rank(design_mat)

        # print some info
        # print('For timestamp '+str(timestamps_cp[i])+':')

        # get the current stations
        stations_here = np.unique(
            np.concatenate((cp_ant1_vec, cp_ant2_vec, cp_ant3_vec))
        )
        # print('Observing stations are '+str([str(station) for station in stations_here]))

        # print('Size of maximal set of closure phases = '+str(ind_here_cp.sum())+'.')
        # print('Size of minimal set of closure phases = '+str(N_min)+'.')
        # print('...')

        ##########################################################
        # start of loop to recover minimal set
        ##########################################################

        # make a mask to keep track of which cphases will stick around
        keep = np.ones(len(obs_cphase_orig), dtype=bool)
        obs_cphase = obs_cphase_orig[keep]

        # remember the original minimal set size
        N_min_orig = N_min

        # initialize the loop breaker
        good_enough = False

        # perform the loop
        count = 0
        ind_list = []
        while good_enough == False:
            # recreate the mask each time
            keep = np.ones(len(obs_cphase_orig), dtype=bool)
            keep[ind_list] = False
            obs_cphase = obs_cphase_orig[keep]

            # organize the closure phase stations
            cp_ant1_vec = obs_cphase["t1"]
            cp_ant2_vec = obs_cphase["t2"]
            cp_ant3_vec = obs_cphase["t3"]

            # get the number of time-matched baselines
            ind_here_bl = time_vis == timestamps_cp[i]
            B_here = ind_here_bl.sum()

            # organize the time-matched baseline stations
            bl_ant1_vec = t1_vis[ind_here_bl]
            bl_ant2_vec = t2_vis[ind_here_bl]

            uvec = obs.data["u"][ind_here_bl]
            vvec = obs.data["v"][ind_here_bl]
            visvec = obs.data["vis"][ind_here_bl]
            sigmavec = obs.data["sigma"][ind_here_bl]

            # initialize the design matrix
            design_mat = np.zeros((keep.sum(), B_here))

            # fill in each row of the design matrix
            for ii in range(keep.sum()):
                # determine which stations are in this triangle
                ant1_here = cp_ant1_vec[ii]
                ant2_here = cp_ant2_vec[ii]
                ant3_here = cp_ant3_vec[ii]

                # matrix entry for first leg of triangle
                ind1_here = (bl_ant1_vec == ant1_here) & (bl_ant2_vec == ant2_here)
                if ind1_here.sum() == 0.0:
                    ind1_here = (bl_ant1_vec == ant2_here) & (bl_ant2_vec == ant1_here)
                    val1_here = -1.0
                else:
                    val1_here = 1.0
                design_mat[ii, ind1_here] = val1_here

                # matrix entry for second leg of triangle
                ind2_here = (bl_ant1_vec == ant2_here) & (bl_ant2_vec == ant3_here)
                if ind2_here.sum() == 0.0:
                    ind2_here = (bl_ant1_vec == ant3_here) & (bl_ant2_vec == ant2_here)
                    val2_here = -1.0
                else:
                    val2_here = 1.0
                design_mat[ii, ind2_here] = val2_here

                # matrix entry for third leg of triangle
                ind3_here = (bl_ant1_vec == ant3_here) & (bl_ant2_vec == ant1_here)
                if ind3_here.sum() == 0.0:
                    ind3_here = (bl_ant1_vec == ant1_here) & (bl_ant2_vec == ant3_here)
                    val3_here = -1.0
                else:
                    val3_here = 1.0
                design_mat[ii, ind3_here] = val3_here

            # determine the size of the minimal set
            N_min = np.linalg.matrix_rank(design_mat)

            if (keep.sum() == N_min_orig) & (N_min == N_min_orig):
                good_enough = True
            else:
                if N_min == N_min_orig:
                    ind_list.append(count)
                else:
                    ind_list = ind_list[:-1]
                    count -= 1
                count += 1

            if count > len(obs_cphase_orig):
                break

        # print out the size of the recovered set for double-checking
        obs_cphase = obs_cphase_orig[keep]
        if len(obs_cphase) != N_min:
            print("*****************WARNING: minimal set not found*****************")
        # else:
        #     print('Size of recovered minimal set = '+str(len(obs_cphase))+'.')
        # print('========================================================================')

        obs_cphase_arr.append(obs_cphase)
        design_mats.append(design_mat)
        uvecs.append(uvec)
        vvecs.append(vvec)
        visvecs.append(visvec)
        sigmavecs.append(sigmavec)

    # save an output cphase file
    obs_cphase_arr = np.concatenate(obs_cphase_arr)
    obs_cphase_arr["cphase"] = obs_cphase_arr["cphase"] * np.pi / 180
    obs_cphase_arr["sigmacp"] = obs_cphase_arr["sigmacp"] * np.pi / 180
    uvec = np.concatenate(uvecs)
    vvec = np.concatenate(vvecs)
    visvec = np.concatenate(visvecs)
    sigmavec = np.concatenate(sigmavecs)

    uvpairs = np.vstack([uvec, vvec]).T
    design_mat = block_diag(design_mats)

    cov = design_mat @ np.diag(sigmavec**2 / np.abs(visvec) ** 2) @ design_mat.T
    return obs_cphase_arr, design_mat, uvpairs, cov


def get_minimal_logcamps(obs, debias=True):
    """
    Given an eht-imaging obs file this will construct the set of minimal log-closure amplitudes that are non-degenerate
    and maximize the SNR.

    The function returns

    - The array of closure phases measured from the data with time, stations, and uv coordinates
    - The sparse design matrix that maps from the set of visibilties in obs (i.e., obs.data["vis"]) to the log-closure amplitude array returned
    - The pairs of the uv-coordinates for each visibility


    This method is adapted from code written by Dom Pesce (see also Blackburn et al, 2021).



    """
    print("Working on getting minimal logcamps...")
    # compute a maximum set of log closure amplitudes
    obs.reorder_tarr_snr()
    obs.add_logcamp(count="max", debias=debias)

    # organize some info
    time_vis = obs.data["time"]
    time_lca = obs.logcamp["time"]
    t1_vis = obs.data["t1"]
    t2_vis = obs.data["t2"]

    # Determine the number of timestamps containing a quadrangle
    timestamps_lca = np.unique(time_lca)
    N_times_lca = len(timestamps_lca)

    # loop over all timestamps
    obs_lca_arr = []
    design_mats = []
    uvecs = []
    vvecs = []
    visvecs = []
    sigmavecs = []

    for i in np.arange(0, N_times_lca, 1):
        # get the current timestamp
        ind_here_lca = time_lca == timestamps_lca[i]
        time_here = time_lca[ind_here_lca]

        # copy the logcamp table for this timestamp
        obs_lca_orig = np.copy(obs.logcamp[ind_here_lca])

        # sort by logcamp SNR
        snr = 1.0 / obs_lca_orig["sigmaca"]
        ind_snr = np.argsort(snr)
        obs_lca_orig = obs_lca_orig[ind_snr]
        snr = snr[ind_snr]

        # organize the quadrangle stations
        lca_ant1_vec = obs_lca_orig["t1"]
        lca_ant2_vec = obs_lca_orig["t2"]
        lca_ant3_vec = obs_lca_orig["t3"]
        lca_ant4_vec = obs_lca_orig["t4"]

        # get the number of time-matched baselines
        ind_here_bl = time_vis == timestamps_lca[i]
        B_here = ind_here_bl.sum()

        # organize the time-matched baseline stations
        bl_ant1_vec = t1_vis[ind_here_bl]
        bl_ant2_vec = t2_vis[ind_here_bl]

        # initialize the design matrix
        design_mat = np.zeros((ind_here_lca.sum(), B_here))

        # fill in each row of the design matrix
        for ii in range(ind_here_lca.sum()):
            # determine which stations are in this quadrangle
            ant1_here = lca_ant1_vec[ii]
            ant2_here = lca_ant2_vec[ii]
            ant3_here = lca_ant3_vec[ii]
            ant4_here = lca_ant4_vec[ii]

            # matrix entry for first leg of quadrangle
            ind1_here = ((bl_ant1_vec == ant1_here) & (bl_ant2_vec == ant2_here)) | (
                (bl_ant1_vec == ant2_here) & (bl_ant2_vec == ant1_here)
            )
            design_mat[ii, ind1_here] = 1.0

            # matrix entry for second leg of quadrangle
            ind2_here = ((bl_ant1_vec == ant3_here) & (bl_ant2_vec == ant4_here)) | (
                (bl_ant1_vec == ant4_here) & (bl_ant2_vec == ant3_here)
            )
            design_mat[ii, ind2_here] = 1.0

            # matrix entry for third leg of quadrangle
            ind3_here = ((bl_ant1_vec == ant1_here) & (bl_ant2_vec == ant4_here)) | (
                (bl_ant1_vec == ant4_here) & (bl_ant2_vec == ant1_here)
            )
            design_mat[ii, ind3_here] = -1.0

            # matrix entry for fourth leg of quadrangle
            ind4_here = ((bl_ant1_vec == ant2_here) & (bl_ant2_vec == ant3_here)) | (
                (bl_ant1_vec == ant3_here) & (bl_ant2_vec == ant2_here)
            )
            design_mat[ii, ind4_here] = -1.0

        # determine the expected size of the minimal set
        N_min = np.linalg.matrix_rank(design_mat)

        # print some info
        # print('For timestamp '+str(timestamps_lca[i])+':')

        # get the current stations
        stations_here = np.unique(
            np.concatenate((lca_ant1_vec, lca_ant2_vec, lca_ant3_vec, lca_ant4_vec))
        )
        # print('Observing stations are '+str([str(station) for station in stations_here]))

        # print('Size of maximal set of closure amplitudes = '+str(ind_here_lca.sum())+'.')
        # print('Size of minimal set of closure amplitudes = '+str(N_min)+'.')
        # print('...')

        ##########################################################
        # start of loop to recover minimal set
        ##########################################################

        # make a mask to keep track of which cphases will stick around
        keep = np.ones(len(obs_lca_orig), dtype=bool)
        obs_lca = obs_lca_orig[keep]

        # remember the original minimal set size
        N_min_orig = N_min

        # initialize the loop breaker
        good_enough = False

        # perform the loop
        count = 0
        ind_list = []
        while good_enough == False:
            # recreate the mask each time
            keep = np.ones(len(obs_lca_orig), dtype=bool)
            keep[ind_list] = False
            obs_lca = obs_lca_orig[keep]

            # organize the quadrangle stations
            lca_ant1_vec = obs_lca["t1"]
            lca_ant2_vec = obs_lca["t2"]
            lca_ant3_vec = obs_lca["t3"]
            lca_ant4_vec = obs_lca["t4"]

            # get the number of time-matched baselines
            ind_here_bl = time_vis == timestamps_lca[i]
            B_here = ind_here_bl.sum()

            # organize the time-matched baseline stations
            bl_ant1_vec = t1_vis[ind_here_bl]
            bl_ant2_vec = t2_vis[ind_here_bl]

            uvec = obs.data["u"][ind_here_bl]
            vvec = obs.data["v"][ind_here_bl]
            visvec = obs.data["vis"][ind_here_bl]
            sigmavec = obs.data["sigma"][ind_here_bl]

            # initialize the design matrix
            design_mat = np.zeros((keep.sum(), B_here))

            # fill in each row of the design matrix
            for ii in range(keep.sum()):
                # determine which stations are in this quadrangle
                ant1_here = lca_ant1_vec[ii]
                ant2_here = lca_ant2_vec[ii]
                ant3_here = lca_ant3_vec[ii]
                ant4_here = lca_ant4_vec[ii]

                # matrix entry for first leg of quadrangle
                ind1_here = (
                    (bl_ant1_vec == ant1_here) & (bl_ant2_vec == ant2_here)
                ) | ((bl_ant1_vec == ant2_here) & (bl_ant2_vec == ant1_here))
                design_mat[ii, ind1_here] = 1.0

                # matrix entry for second leg of quadrangle
                ind2_here = (
                    (bl_ant1_vec == ant3_here) & (bl_ant2_vec == ant4_here)
                ) | ((bl_ant1_vec == ant4_here) & (bl_ant2_vec == ant3_here))
                design_mat[ii, ind2_here] = 1.0

                # matrix entry for third leg of quadrangle
                ind3_here = (
                    (bl_ant1_vec == ant1_here) & (bl_ant2_vec == ant4_here)
                ) | ((bl_ant1_vec == ant4_here) & (bl_ant2_vec == ant1_here))
                design_mat[ii, ind3_here] = -1.0

                # matrix entry for fourth leg of quadrangle
                ind4_here = (
                    (bl_ant1_vec == ant2_here) & (bl_ant2_vec == ant3_here)
                ) | ((bl_ant1_vec == ant3_here) & (bl_ant2_vec == ant2_here))
                design_mat[ii, ind4_here] = -1.0

            # determine the size of the minimal set
            N_min = np.linalg.matrix_rank(design_mat)

            if (keep.sum() == N_min_orig) & (N_min == N_min_orig):
                good_enough = True
            else:
                if N_min == N_min_orig:
                    ind_list.append(count)
                else:
                    ind_list = ind_list[:-1]
                    count -= 1
                count += 1

            if count > len(obs_lca_orig):
                break

        # print out the size of the recovered set for double-checking
        obs_lca = obs_lca_orig[keep]
        if len(obs_lca) != N_min:
            print("*****************WARNING: minimal set not found*****************")
        # else:
        #     print('Size of recovered minimal set = '+str(len(obs_lca))+'.')
        # print('========================================================================')

        obs_lca_arr.append(obs_lca)
        design_mats.append(design_mat)
        uvecs.append(uvec)
        vvecs.append(vvec)
        visvecs.append(visvec)
        sigmavecs.append(sigmavec)

    # save an output logcamp file
    obs_lca_arr = np.concatenate(obs_lca_arr)
    uvec = np.concatenate(uvecs)
    vvec = np.concatenate(vvecs)
    visvec = np.concatenate(visvecs)
    sigmavec = np.concatenate(sigmavecs)
    uvpairs = np.vstack([uvec, vvec]).T
    design_mat = block_diag(design_mats)

    cov = design_mat @ np.diag(sigmavec**2 / np.abs(visvec) ** 2) @ design_mat.T

    return obs_lca_arr, design_mat, uvpairs, cov
