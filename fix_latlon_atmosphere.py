#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from functools import reduce
import logging
import math
import operator
import os
import re

import iris
import netCDF4
import numpy as np


logging.basicConfig()


def binlist(n, width=0):
    """Return list of bits that represent a non-negative integer.

    n      -- non-negative integer
    width  -- number of bits in returned zero-filled list (default 0)
    """
    return map(int, list(bin(n)[2:].zfill(width)))


def numVals(shape):
    """Return number of values in chunk of specified shape, given by a list of dimension lengths.

    shape -- list of variable dimension sizes"""
    if(len(shape) == 0):
        return 1
    return reduce(operator.mul, shape)


def perturbShape(shape, onbits):
    """Return shape perturbed by adding 1 to elements corresponding to 1 bits in onbits

    shape  -- list of variable dimension sizes
    onbits -- non-negative integer less than 2**len(shape)
    """
    return list(map(sum, zip(shape, binlist(onbits, len(shape)))))


def chunk_shape_3D(varShape, valSize=4, chunkSize=4096):
    """
    Return a 'good shape' for a 3D variable, assuming balanced 1D/(n-1)D access

    varShape  -- length 3 list of variable dimension sizes
    chunkSize -- maximum chunksize desired, in bytes (default 4096)
    valSize   -- size of each data value, in bytes (default 4)

    Returns integer chunk lengths of a chunk shape that provides
    balanced access of 1D subsets and 2D subsets of a netCDF or HDF5
    variable var with shape (T, X, Y), where the 1D subsets are of the
    form var[:,x,y] and the 2D slices are of the form var[t,:,:],
    typically 1D time series and 2D spatial slices.  'Good shape' for
    chunks means that the number of chunks accessed to read either
    kind of 1D or 2D subset is approximately equal, and the size of
    each chunk (uncompressed) is no more than chunkSize, which is
    often a disk block size.
    """

    rank = 3
    chunkVals = chunkSize / float(valSize) # ideal number of values in a chunk
    numChunks  = varShape[0]*varShape[1]*varShape[2] / chunkVals # ideal number of chunks
    axisChunks = numChunks ** 0.25 # ideal number of chunks along each 2D axis
    cFloor = [] # will be first estimate of good chunk shape
    # cFloor  = [varShape[0] // axisChunks**2, varShape[1] // axisChunks, varShape[2] // axisChunks]
    # except that each chunk shape dimension must be at least 1
    # chunkDim = max(1.0, varShape[0] // axisChunks**2)
    if varShape[0] / axisChunks**2 < 1.0:
        chunkDim = 1.0
        axisChunks = axisChunks / math.sqrt(varShape[0]/axisChunks**2)
    else:
        chunkDim = varShape[0] // axisChunks**2
    cFloor.append(chunkDim)
    prod = 1.0  # factor to increase other dims if some must be increased to 1.0
    for i in range(1, rank):
        if varShape[i] / axisChunks < 1.0:
            prod *= axisChunks / varShape[i]
    for i in range(1, rank):
        if varShape[i] / axisChunks < 1.0:
            chunkDim = 1.0
        else:
            chunkDim = (prod*varShape[i]) // axisChunks
        cFloor.append(chunkDim)

    # cFloor is typically too small, (numVals(cFloor) < chunkSize)
    # Adding 1 to each shape dim results in chunks that are too large,
    # (numVals(cCeil) > chunkSize).  Want to just add 1 to some of the
    # axes to get as close as possible to chunkSize without exceeding
    # it.  Here we use brute force, compute numVals(cCand) for all
    # 2**rank candidates and return the one closest to chunkSize
    # without exceeding it.
    bestChunkSize = 0
    cBest = cFloor
    for i in range(8):
        # cCand = map(sum,zip(cFloor, binlist(i, rank)))
        cCand = perturbShape(cFloor, i)
        thisChunkSize = valSize * numVals(cCand)
        if bestChunkSize < thisChunkSize <= chunkSize:
            bestChunkSize = thisChunkSize
            cBest = list(cCand) # make a copy of best candidate so far
    return list(map(int, cBest))


def needs_fix(cube):
    if cube.attributes['realm'] != 'atmos':
        return False
    if cube.coord('latitude').ndim > 1 or cube.coord('longitude').ndim > 1:
        return True
    return False


def bounds_enclose_points_1d(bounds, points):
    return (bounds[:, 0] < points).all() and (points < bounds[:, 1]).all()


def coord_2d_to_1d(coord):
    kind = coord.standard_name
    if kind == 'latitude':
        axis = 1
        corners = (0, 3)
        var_name = 'lat'
        circular = False
    elif kind == 'longitude':
        axis = 0
        corners = (0, 1)
        var_name = 'lon'
        circular = True
    else:
        raise ValueError("coord must have standard_name 'latitude' or 'longitude'.")
    assert np.alltrue(np.diff(coord.points, axis=axis)==0.)
    points_1d = np.take(coord.points, 0, axis)
    bounds_1d = np.take(coord.bounds, 0, axis)[:, corners]
    if not bounds_enclose_points_1d(bounds_1d, points_1d):
        logging.warning("Bounds are probably in wrong order, correcting.")
        bounds_1d = bounds_1d[::-1, ::-1]
    if not bounds_enclose_points_1d(bounds_1d, points_1d):
        logging.error("Bounds are still incorrect. Giving up.")
        raise RuntimeError("Uncorrectable bounds for {}".format(kind))
    logging.info("Bounds are correct now.")
    coord_1d = iris.coords.DimCoord(
        points_1d,
        standard_name=coord.standard_name,
        long_name=coord.long_name,
        var_name=var_name,
        units=coord.units,
        bounds=bounds_1d,
        attributes=coord.attributes,
        coord_system=coord.coord_system,
        circular=circular)
    return coord_1d


def remove_superfluous_dim_coords(cube, dims):
    expr = re.compile('cell index along (first|second) dimension')
    for dim in dims:
        try:
            superfluous_dim_coord = cube.coord(dimensions=dim, dim_coords=True)
        except iris.exceptions.CoordinateNotFoundError:
            pass
        else:
            long_name = superfluous_dim_coord.long_name
            if expr.match(long_name) is None:
                raise RuntimeError("Found dim coord on horizontal dimensions "
                                   "with unexpected long_name '{}'".format(
                                       long_name))
            cube.remove_coord(superfluous_dim_coord)


def replace_2d_coords_with_1d(cube):
    latitude = cube.coord('latitude')
    lat = coord_2d_to_1d(latitude)
    longitude = cube.coord('longitude')
    lon = coord_2d_to_1d(longitude)
    dims = cube.coord_dims(latitude)
    assert dims == cube.coord_dims('longitude')
    cube.remove_coord(latitude)
    cube.remove_coord(longitude)
    remove_superfluous_dim_coords(cube, dims)
    cube.add_dim_coord(lat, dims[0])
    cube.add_dim_coord(lon, dims[1])
    return cube


def fix_attributes(original_filename, tmp_filename):
    ds_o = netCDF4.Dataset(original_filename)
    ds_n = netCDF4.Dataset(tmp_filename, 'r+')
    var = os.path.basename(original_filename).split('_')[0]
    assert var == os.path.basename(tmp_filename).split('_')[0]
    v_o = ds_o[var]
    v_n = ds_n[var]
    ds_n.history = ds_o.history
    ds_n.Conventions = ds_o.Conventions
    v_n.history = v_o.history
    v_n.cell_measures = v_o.cell_measures
    if hasattr(v_o, 'missing_value'):
        v_n.missing_value = v_o.missing_value
    ds_n.close()
    ds_o.close()


def perform_fix(original_filename, cube, chunk_size):
    cube = replace_2d_coords_with_1d(cube)
    tmp_filename = original_filename+"_new.nc"
    shape = cube.shape
    val_size = cube.dtype.itemsize
    if len(shape) == 3:
        chunksizes = chunk_shape_3D(shape, val_size, chunk_size)
    elif len(shape) == 4:
        chunksizes = chunk_shape_3D((shape[0],) + shape[2:],
                                    val_size, chunk_size)
        chunksizes.insert(1, 1)
    else:
        chunksizes = None
    iris.save(cube, tmp_filename,
              unlimited_dimensions=['time'],
              zlib=True, complevel=3, shuffle=True,
              chunksizes=chunksizes,
              fill_value=1.e20)
    fix_attributes(original_filename, tmp_filename)
    backup_filename = original_filename+"_backup.nc"
    os.rename(original_filename, backup_filename)
    os.rename(tmp_filename, original_filename)
    os.remove(backup_filename)


def binary_size(string):
    SUFFIX_FACTORS = {
        'B': 1,
        'KiB': 1024,
        'MiB': 1024**2,
        'GiB': 1024**3,
        'TiB': 1024**4,
        'PiB': 1024**5,
        'EiB': 1024**6,
        'ZiB': 1024**7,
        'YiB': 1024**8,
    }
    valid = re.compile(r'(?P<value>\d*\.?\d*) *(?P<suffix>([KMGTPEZY]i)?B)')
    match = valid.match(string.strip())
    if match is None:
        raise argparse.ArgumentTypeError(
            "Invalid size argument {}. Must be a number followed by either B "
            "for bytes or one of the IEC units KiB, MiB, etc.".format(string))
    num = float(match.group('value'))
    suffix = match.group('suffix')
    factor = SUFFIX_FACTORS[suffix]
    size = num*factor
    return size


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--chunk-size',
                        type=binary_size, default='64KiB',
                        help='Target chunk size in terms of bytes. '
                        'Must be given with units, either B for bytes '
                        'or one of the IEC units (KiB, MiB, ...)')
    parser.add_argument('file')
    return parser.parse_args()


def main():
    args = parse_args()
    original_filename = args.file
    cube = iris.load_cube(original_filename)
    if needs_fix(cube):
        perform_fix(original_filename, cube, args.chunk_size)
    else:
        logging.info("No fix needed or fix not applicable.")


if __name__ == '__main__':
    main()
