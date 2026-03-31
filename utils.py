#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nibabel as nib
import numpy as np

from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.tracking.metrics import length as streamline_length
from dipy.tracking.streamline import set_number_of_points
from dipy.tracking.vox2track import streamline_mapping

from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree


def loadTrk(filename):
    data = nib.streamlines.load(filename)
    s = data.streamlines
    aff = data.affine
    header = data.header
    return s,aff,header


def saveTrackDipy(track, output_file,
                  structural_filename=None,
                  remove_invalid_streamlines=True,
                  header=None,
                  bbox_valid_check=True):

    # --------------------------------------------------
    # Resolve reference image
    # --------------------------------------------------
    if structural_filename is not None:
        struct_nib = nib.load(structural_filename)

    elif header is not None:
        struct_nib = header

    else:
        raise ValueError(
            "saveTrackDipy(): no reference provided.\n"
            "You must supply either:\n"
            "  - structural_filename (path to NIfTI), or\n"
            "  - header/reference object (Nifti1Image or compatible).\n"
            f"Output file: {output_file}"
        )

    # --------------------------------------------------
    # Build StatefulTractogram
    # --------------------------------------------------
    track_sft = StatefulTractogram(track, struct_nib, Space.RASMM)

    if remove_invalid_streamlines:
        track_sft.remove_invalid_streamlines()

    print(f"bbox_valid_check: {bbox_valid_check}")

    save_tractogram(
        track_sft,
        output_file,
        bbox_valid_check=bbox_valid_check
    )



def gkernel(l=3, sig=2):
    """\
    Gaussian Kernel Creator via given length and sigma
    """

    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy, zz = np.meshgrid(ax, ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)+ np.square(zz)) / np.square(sig))

    return kernel / np.sum(kernel)

def streamlines_count(track,track_aff,dimensions,stream_map=None,smooth_density=True):
    if stream_map is None:
        stream_map = streamline_mapping(track,affine=track_aff)
    streamline_count = np.zeros(dimensions)
    for idx in list(stream_map.keys()):
         streamline_count[idx] = len(stream_map[idx])
    if smooth_density:
        print('smooth density')		
        from scipy import signal
        kernel=gkernel(l=4, sig=2)
        streamline_count = signal.fftconvolve(streamline_count, kernel, mode='same')

    return streamline_count
def get_core_streamlines(track_filename, perc=0.75, output_file=None, structural_filename=None,smooth_density=True):
    
    # %%load data	
    track,track_aff,track_header = loadTrk(track_filename)
    # %%calculate density map
    print("calculate density map...")
    # --- Safe tractogram loading ---
    if track_filename.endswith(".trk"):
        # .trk files contain their own reference info
        sft = load_tractogram(track_filename, "same",
                            to_space=Space.RASMM,
                            bbox_valid_check=True)
        #track_aff=sft.affine
        print("Affine transformation will be taken directly from the TRK file header.")
    else:
        # for .tck files, use the provided reference (if any)
        if structural_filename is not None:
            ref = nib.load(structural_filename)
        else:
            # fall back to first ROI or raise error
            raise ValueError(
                "Reference image (structural_filename) must be provided for .tck files."
            )
        sft = load_tractogram(track_filename, ref,
                            to_space=Space.RASMM,
                            bbox_valid_check=True)
        track_aff=ref.affine
        print("Affine transformation will be taken from the reference NIfTI image for spatial alignment.")
    print(track_aff)
    #track = sft.streamlines
    num_streamlines=len(track)
    print("number of streamlines: "+str(num_streamlines))
    sft.to_vox()
    sft.to_corner()
    transformation, dimensions, _, _ = sft.space_attributes
    stream_map = streamline_mapping(track,affine=track_aff)
    streamline_count_array = streamlines_count(track,track_aff,dimensions,stream_map,smooth_density=smooth_density)
    #perc = np.percentile(np.unique(streamline_count_array), 75)
    #percent = np.percentile(streamline_count_array[streamline_count_array>0.5*np.max(streamline_count_array)],perc)
    #indices = np.where(streamline_count_array > percent )
    
     
    if perc != 0:
        indices = np.where(streamline_count_array >= perc*np.max(streamline_count_array) )
    else:
        indices = np.where(streamline_count_array > perc*np.max(streamline_count_array) )
    streamlines_to_keep = []
    streamlines_to_keep_dic = dict()
    for i in range(len(indices[0])):
        idx = (indices[0][i],indices[1][i],indices[2][i])
        for element in stream_map[idx]:
                streamlines_to_keep.append(element)
                try:
                      streamlines_to_keep_dic[element] = streamlines_to_keep_dic[element] + 1
                except:
                      streamlines_to_keep_dic[element] =  1
    streamlines_to_keep=np.unique(streamlines_to_keep)
    track = track[streamlines_to_keep]
    num_streamlines=len(track)
    print("number of cores streamlines: "+str(num_streamlines))
    if  output_file is not None:
        saveTrackDipy(track, output_file, structural_filename=structural_filename, remove_invalid_streamlines=False)
 
    return track



def get_bundle_backbone(track_filename, output_file, structural_filename, 
                        N_points=32, perc=0.75, smooth_density=True, 
                        length_thr=0.9,
                        keep_endpoints=False,
                        average_type="mean",
                        endpoint_mode="median",       # "mean", "median", "median_project"
                        representative=False,         # choose closest streamline instead of backbone
                        spline_smooth=None):          # apply spline smoothing at the end
    """
    Compute backbone of a streamline bundle, optionally extract representative streamline,
    and apply optional spline smoothing at the end.

    endpoint_mode:
        "mean"            → endpoints = mean(coords)
        "median"          → endpoints = median(coords)
        "median_project"  → project median point to closest actual streamline endpoint
    """

    # ----------------------------------------------------------
    # 1. Load & preprocess bundle
    # ----------------------------------------------------------
    if perc != 0:
        print("get core streamlines...")
        track = get_core_streamlines(
            track_filename,
            structural_filename=structural_filename,
            perc=perc,
            smooth_density=smooth_density
        )
    else:
        track, track_aff, track_header = loadTrk(track_filename)

    print("Resampling streamlines to N_points =", N_points)
    track = set_number_of_points(track, N_points)

    # ----------------------------------------------------------
    # 2. Orient streamlines consistently
    # ----------------------------------------------------------
    streamline_0 = track[0]
    length = np.zeros(len(track))

    for i, sl in enumerate(track):
        dist      = np.linalg.norm((streamline_0 - sl).reshape(-1, N_points*3), axis=1)
        dist_flip = np.linalg.norm((streamline_0 - sl[::-1]).reshape(-1, N_points*3), axis=1)

        if dist_flip < dist:
            sl = sl[::-1]

        track[i] = sl
        length[i] = streamline_length(sl)

    # keep only long-enough streamlines
    length_mask = length > (length_thr * np.max(length))
    track = track[length_mask]

    # ----------------------------------------------------------
    # 3. Compute backbone (mean/median)
    # ----------------------------------------------------------
    backbone = []
    num_points = len(track[0])

    for p in range(num_points):
        coords = np.array([sl[p] for sl in track])

        # ---- Endpoint logic ----
        if keep_endpoints and (p == 0 or p == num_points - 1):

            if endpoint_mode == "mean":
                agg = np.mean(coords, axis=0)

            elif endpoint_mode == "median":
                agg = np.median(coords, axis=0)

            elif endpoint_mode == "median_project":
                med = np.median(coords, axis=0)
                endpoints = np.array([sl[p] for sl in track])
                d = np.linalg.norm(endpoints - med, axis=1)
                agg = endpoints[d.argmin()]

            else:
                raise ValueError(f"Unknown endpoint_mode: {endpoint_mode}")

        # ---- Interior points ----
        else:
            agg = np.median(coords, axis=0) if average_type == "median" else np.mean(coords, axis=0)

        backbone.append(agg)

    backbone = np.array(backbone)  # shape: (N_points, 3)

    # ----------------------------------------------------------
    # 4. If representative mode → choose closest streamline
    # ----------------------------------------------------------
    if representative:
        print("Selecting representative streamline (closest to backbone).")

        tree = cKDTree(np.array(track).reshape(len(track), -1))
        dist, idx = tree.query(backbone.reshape(-1), k=1)
        rep = track[idx]

        final = rep.copy()

    else:
        final = backbone.copy()

    # ----------------------------------------------------------
    # 5. Apply spline smoothing LAST
    # ----------------------------------------------------------
    if spline_smooth is not None:
        print(f"Applying spline smoothing with s={spline_smooth}")

        tck, u = splprep(final.T, s=float(spline_smooth))
        u_new = np.linspace(0, 1, num_points)
        x, y, z = splev(u_new, tck)
        smoothed = np.vstack([x, y, z]).T

        # preserve endpoints if requested (projected/median already computed)
        if keep_endpoints:
            smoothed[0]  = final[0]
            smoothed[-1] = final[-1]

        final = smoothed

    # reshape & save
    final = np.array([final])
    saveTrackDipy(final, output_file, structural_filename=structural_filename)

    return final


def get_bundle_backbone_from_streamlines(
    track,
    track_aff,
    dimensions,
    N_points=32,
    perc=0,
    smooth_density=True,
    length_thr=0.9,
    keep_endpoints=False,
    average_type="mean",
    endpoint_mode="median",
    representative=False,
    spline_smooth=None,
    verbose=True
):
    """
    In-memory backbone with optional core-streamline extraction.
    Returns: np.ndarray of shape (1, N_points, 3)
    """

    # ----------------------------------------------------------
    # 1. Core streamline extraction
    # ----------------------------------------------------------
    track = _ensure_streamlines(track)

    if perc != 0:
        if verbose:
            print("get core streamlines...")

        track = get_core_streamlines_from_streamlines(
            track,
            track_aff,
            dimensions,
            perc=perc,
            smooth_density=smooth_density,
            verbose=verbose
        )
        track = _ensure_streamlines(track)

    if len(track) == 0:
        return None

    # ----------------------------------------------------------
    # 2. Resample 
    # ----------------------------------------------------------
    track = Streamlines([_as_float32_streamline(sl) for sl in track if _as_float32_streamline(sl) is not None])
    track = set_number_of_points(track, N_points)

    # after resample, all should be same length -> stack into real float array
    track = np.asarray(track, dtype=np.float32)      # (M, N_points, 3) float32

    # ----------------------------------------------------------
    # 3. Orientation + length filtering
    # ----------------------------------------------------------
    ref = track[0]
    lengths = np.zeros(len(track))

    for i, sl in enumerate(track):
        d = np.linalg.norm(ref - sl)
        d_flip = np.linalg.norm(ref - sl[::-1])
        if d_flip < d:
            track[i] = sl[::-1]
        diff = np.diff(track[i], axis=0)          # (N-1,3) float32
        lengths[i] = np.linalg.norm(diff, axis=1).sum()


    keep = lengths > (length_thr * np.max(lengths))
    track = track[keep]

    if len(track) == 0:
        return None

    # ----------------------------------------------------------
    # 4. Backbone computation
    # ----------------------------------------------------------
    num_points = track.shape[1]
    backbone = np.zeros((num_points, 3))

    for p in range(num_points):
        coords = np.array([sl[p] for sl in track])

        if keep_endpoints and (p == 0 or p == num_points - 1):
            if endpoint_mode == "mean":
                backbone[p] = np.mean(coords, axis=0)
            elif endpoint_mode == "median":
                backbone[p] = np.median(coords, axis=0)
            elif endpoint_mode == "median_project":
                med = np.median(coords, axis=0)
                backbone[p] = coords[np.linalg.norm(coords - med, axis=1).argmin()]
            else:
                raise ValueError(endpoint_mode)
        else:
            backbone[p] = (
                np.median(coords, axis=0)
                if average_type == "median"
                else np.mean(coords, axis=0)
            )

    # ----------------------------------------------------------
    # 5. Representative (SAFE)
    # ----------------------------------------------------------
    if representative:
        from scipy.spatial import cKDTree
        tree = cKDTree(track.reshape(len(track), -1))
        _, idx = tree.query(backbone.reshape(-1), k=1)
        final = track[idx]              # (N,3)
    else:
        final = backbone                # (N,3)

    # ----------------------------------------------------------
    # 6. Spline smoothing
    # ----------------------------------------------------------
    if spline_smooth is not None:
        tck, _ = splprep(final.T, s=float(spline_smooth))
        u = np.linspace(0, 1, num_points)
        x, y, z = splev(u, tck)
        smoothed = np.vstack([x, y, z]).T

        if keep_endpoints:
            smoothed[0] = final[0]
            smoothed[-1] = final[-1]

        final = smoothed

    return final[np.newaxis, :, :]
