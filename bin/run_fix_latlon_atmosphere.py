#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging

from highresmip_fix.fix_latlon_atmosphere import (binary_size,
                                                  fix_latlon_atmosphere)

logging.basicConfig()


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
    fix_latlon_atmosphere(args.file, args.chunk_size)


if __name__ == '__main__':
    main()
