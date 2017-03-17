#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import os
import re
import sys
import shlex
import json
import threading
import webbrowser
from datetime import datetime
from pathlib import Path
from subprocess import check_call, check_output, CalledProcessError, DEVNULL

import click
import requests
from osgeo.osr import SpatialReference

from lopocs import __version__
from lopocs import create_app, greyhound, threedtiles
from lopocs.database import Session
from lopocs.potreeschema import potree_schema

# intialize flask application
app = create_app()

samples = {
    'airport': 'http://www.liblas.org/samples/LAS12_Sample_withRGB_Quick_Terrain_Modeler_fixed.las',
    'stsulpice': 'https://freefr.dl.sourceforge.net/project/e57-3d-imgfmt/E57Example-data/Trimble_StSulpice-Cloud-50mm.e57'
}


def fatal(message):
    '''print error and exit'''
    click.echo('\nFATAL: {}'.format(message), err=True)
    sys.exit(1)


def pending(msg, nl=False):
    click.echo('[{}] {} ... '.format(
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        msg
    ), nl=nl)


def ok(mess=None):
    click.secho('ok: {}'.format(mess) if mess else 'ok', fg='green')


def ko():
    click.secho('ko', fg='red')


def print_version(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    click.echo('LOPoCS version {}'.format(__version__))
    click.echo('')
    ctx.exit()


@click.group()
@click.option('--version', help='show version', is_flag=True, expose_value=False, callback=print_version)
def cli():
    '''lopocs command line tools'''
    pass


@cli.command()
def serve():
    '''run lopocs server (development usage)'''
    app.run()


@click.option('--table', required=True, help='table name to store pointclouds, considered in public schema if no prefix provided')
@click.option('--column', help="column name to store patches", default="points", type=str)
@click.option('--work-dir', type=click.Path(exists=True), required=True, help="working directory where temporary files will be saved")
@click.option('--server-url', type=str, help="server url for lopocs", default="http://localhost:5000")
@click.argument('filename', type=click.Path(exists=True))
@cli.command()
def load(filename, table, column, work_dir, server_url):
    '''load pointclouds data using pdal and add metadata needed by lopocs'''
    _load(filename, table, column, work_dir, server_url)


def _load(filename, table, column, work_dir, server_url):
    '''load pointclouds data using pdal and add metadata needed by lopocs'''
    filename = Path(filename)
    work_dir = Path(work_dir)
    extension = filename.suffix[1:].lower()
    basename = filename.stem
    basedir = filename.parent

    pending('Creating metadata table')
    Session.create_pointcloud_lopocs_table()
    ok()

    pending('Loading point clouds into database')
    json_path = os.path.join(
        str(work_dir.resolve()),
        '{basename}_{table}_pipeline.json'.format(**locals()))

    # tablename should be always prefixed
    if '.' not in table:
        table = 'public.{}'.format(table)

    cmd = "pdal info --summary {}".format(filename)
    try:
        output = check_output(shlex.split(cmd))
    except CalledProcessError as e:
        fatal(e)

    summary = json.loads(output.decode())['summary']

    if summary['srs']['isgeographic']:
        # geographic
        scale_x, scale_y, scale_z = (1e-6, 1e-6, 1e-2)
    else:
        # projection or geocentric
        scale_x, scale_y, scale_z = (0.01, 0.01, 0.01)

    offset_x = summary['bounds']['X']['min'] + (summary['bounds']['X']['max'] - summary['bounds']['X']['min']) / 2
    offset_y = summary['bounds']['Y']['min'] + (summary['bounds']['Y']['max'] - summary['bounds']['Y']['min']) / 2
    offset_z = summary['bounds']['Z']['min'] + (summary['bounds']['Z']['max'] - summary['bounds']['Z']['min']) / 2

    offset_x = round(offset_x, 2)
    offset_y = round(offset_y, 2)
    offset_z = round(offset_z, 2)

    if extension == 'e57':
        # summary gives empty results for this format
        offset_x = 0
        offset_y = 0
        offset_z = 0
        scale_x, scale_y, scale_z = (1, 1, 1)

    pg_name = app.config['PG_NAME']
    pg_port = app.config['PG_PORT']
    pg_user = app.config['PG_USER']
    pg_password = app.config['PG_PASSWORD']
    realfilename = str(filename.resolve())
    schema, tab = table.split('.')
    srid = proj42epsg(summary['srs']['proj4'])

    json_pipeline = """
{{
"pipeline": [
    {{
        "type": "readers.{extension}",
        "filename":"{realfilename}"
    }},
    {{
        "type": "filters.chipper",
        "capacity":400
    }},
    {{
        "type": "filters.revertmorton"
    }},
    {{
        "type":"writers.pgpointcloud",
        "connection":"dbname={pg_name} port={pg_port} user={pg_user} password={pg_password}",
        "schema": "{schema}",
        "table":"{tab}",
        "compression":"none",
        "srid":"{srid}",
        "overwrite":"true",
        "column": "points",
        "scale_x": "{scale_x}",
        "scale_y": "{scale_y}",
        "scale_z": "{scale_z}",
        "offset_x": "{offset_x}",
        "offset_y": "{offset_y}",
        "offset_z": "{offset_z}"
    }}
]
}}""".format(**locals())

    with io.open(json_path, 'w') as json_file:
        json_file.write(json_pipeline)

    cmd = "pdal pipeline {}".format(json_path)

    try:
        check_call(shlex.split(cmd), stderr=DEVNULL, stdout=DEVNULL)
    except CalledProcessError as e:
        fatal(e)
    ok()

    pending("Creating indexes")
    Session.execute("""
        create index on {table} using gist(geometry(points));
        alter table {table} add column morton bigint;
        select Morton_Update('{table}', 'points', 'morton', 64, TRUE);
        create index on {table}(morton);
    """.format(**locals()))
    ok()

    pending("Adding metadata for lopocs")
    Session.update_metadata(
        table, column, srid, scale_x, scale_y, scale_z,
        offset_x, offset_y, offset_z
    )
    # add schema currently used by potree (version 1.5RC)
    Session.add_output_schema(
        table, column, 0.01, 0.01, 0.01,
        offset_x, offset_y, offset_z, srid, potree_schema
    )
    lpsession = Session(table, column)
    ok()

    # initialize range for level of details
    lod_min = 0
    lod_max = 5

    # retrieve boundingbox
    fullbbox = lpsession.boundingbox
    bbox = [
        fullbbox['xmin'], fullbbox['ymin'], fullbbox['zmin'],
        fullbbox['xmax'], fullbbox['ymax'], fullbbox['zmax']
    ]
    cache_file = (
        "{0}_{1}_{2}_{3}_{4}.hcy".format(
            lpsession.table,
            lpsession.column,
            lod_min,
            lod_max,
            '_'.join(str(e) for e in bbox)
        )
    )

    pending("Building greyhound hierarchy")
    new_hcy = greyhound.build_hierarchy_from_pg(
        lpsession, lod_min, lod_max, bbox
    )
    greyhound.write_in_cache(new_hcy, cache_file)
    ok()

    pending("Building 3Dtiles tileset")
    hcy = threedtiles.build_hierarchy_from_pg(
        lpsession, server_url, lod_max, bbox, lod_min
    )

    tileset = os.path.join(str(work_dir.resolve()), 'tileset.json')

    with io.open(tileset, 'wb') as out:
        out.write(hcy.encode())
    ok()


@cli.command()
@click.option('--sample', help="sample lidar file to test", default="airport", type=click.Choice(['airport', 'stsulpice']))
@click.option('--work-dir', type=click.Path(exists=True), required=True, help="working directory where sample files will be saved")
@click.option('--server-url', type=str, help="server url for lopocs", default="http://localhost:5000")
def demo(sample, work_dir, server_url):
    '''
    Download sample lidar data, load it into your and visualize in potree viewer !
    '''
    filepath = Path(samples[sample])
    dest = os.path.join(work_dir, filepath.name)

    if not os.path.exists(dest):
        r = requests.get(samples[sample], stream=True)
        length = int(r.headers['content-length'])

        chunk_size = 512
        iter_size = 0

        with io.open(dest, 'wb') as fd:
            with click.progressbar(length=length, label='Downloading sample file') as bar:
                for chunk in r.iter_content(chunk_size):
                    fd.write(chunk)
                    iter_size += chunk_size
                    bar.update(chunk_size)

    # now load data
    _load(dest, sample, 'points', work_dir, server_url)
    # open tab to the API
    threading.Timer(1.5, lambda: webbrowser.open_new_tab(server_url)).start()
    # run application
    app.run(debug=False)


def proj42epsg(proj4, epsg='/usr/share/proj/epsg', forceProj4=False):
    ''' Transform a WKT string to an EPSG code

    Arguments
    ---------

    proj4: proj4 string definition
    epsg: the proj.4 epsg file (defaults to '/usr/local/share/proj/epsg')
    forceProj4: whether to perform brute force proj4 epsg file check (last resort)

    Returns: EPSG code

    '''
    code = '4326'
    p_in = SpatialReference()
    s = p_in.ImportFromProj4(proj4)
    if s == 5:  # invalid WKT
        return '%s' % code
    if p_in.IsLocal() == 1:  # this is a local definition
        return p_in.ExportToWkt()
    if p_in.IsGeographic() == 1:  # this is a geographic srs
        cstype = 'GEOGCS'
    else:  # this is a projected srs
        cstype = 'PROJCS'
    an = p_in.GetAuthorityName(cstype)
    ac = p_in.GetAuthorityCode(cstype)
    if an is not None and ac is not None:  # return the EPSG code
        return '%s' % p_in.GetAuthorityCode(cstype)
    else:  # try brute force approach by grokking proj epsg definition file
        p_out = p_in.ExportToProj4()
        if p_out:
            if forceProj4 is True:
                return p_out
            f = open(epsg)
            for line in f:
                if line.find(p_out) != -1:
                    m = re.search('<(\\d+)>', line)
                    if m:
                        code = m.group(1)
                        break
            if code:  # match
                return '%s' % code
            else:  # no match
                return '%s' % code
        else:
            return '%s' % code