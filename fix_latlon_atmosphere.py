#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import os

import iris
import netCDF4
import numpy as np


logging.basicConfig()


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


def replace_2d_coords_with_1d(cube):
    latitude = cube.coord('latitude')
    lat = coord_2d_to_1d(latitude)
    longitude = cube.coord('longitude')
    lon = coord_2d_to_1d(longitude)
    dims = cube.coord_dims(latitude)
    assert dims == cube.coord_dims('longitude')
    cube.remove_coord(latitude)
    cube.remove_coord(longitude)
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


def perform_fix(original_filename, cube):
    cube = replace_2d_coords_with_1d(cube)
    tmp_filename = original_filename+"_new.nc"
    iris.save(cube, tmp_filename, fill_value=1.e20)
    fix_attributes(original_filename, tmp_filename)
    backup_filename = original_filename+"_backup.nc"
    os.rename(original_filename, backup_filename)
    os.rename(tmp_filename, original_filename)
    os.remove(backup_filename)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    return parser.parse_args()


def main():
    args = parse_args()
    original_filename = args.file
    cube = iris.load_cube(original_filename)
    if needs_fix(cube):
        perform_fix(original_filename, cube)
    else:
        logging.info("No fix needed or fix not applicable.")


if __name__ == '__main__':
    main()
