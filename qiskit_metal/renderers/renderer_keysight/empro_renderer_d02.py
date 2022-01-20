# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import math
from qiskit_metal import Dict
from scipy.spatial import distance
import os
import geopandas
import shapely

from shapely.geometry import LineString as LineString
from copy import deepcopy
from operator import itemgetter
from typing import TYPE_CHECKING
from typing import Dict as Dict_
from typing import List, Tuple, Union, Any, Iterable
import pandas as pd
from pandas.api.types import is_numeric_dtype

import numpy as np

from qiskit_metal.renderers.renderer_base import QRenderer
from qiskit_metal.toolbox_metal.parsing import is_true

from qiskit_metal import config
if not config.is_building_docs():
    from qiskit_metal.toolbox_python.utility_functions import can_write_to_path
    from qiskit_metal.toolbox_python.utility_functions import get_range_of_vertex_to_not_fillet

if TYPE_CHECKING:
    # For linting typechecking, import modules that can't be loaded here under normal conditions.
    # For example, I can't import QDesign, because it requires Qrenderer first. We have the
    # chicken and egg issue.
    from qiskit_metal.designs import QDesign


class QEMProRenderer(QRenderer):
    """Extends QRenderer to create new EMPro QRenderer.

    This QRenderer will print to a file the number_of_bones and the
    names of QGeometry tables that will be used to export the
    QComponents the user highlighted.
    """

    #: Default options, over-written by passing ``options` dict to render_options.
    #: Type: Dict[str, str]
    default_options = Dict(
        # An option unique to QEMProRenderer.
        height='0.10 nm',
        # Default file name for geometry table names.
        file_geometry_tables='./simple_output_default_name.txt')
    """Default options"""

    name = 'empro'
    """Name used in Metal code to refer to this QRenderer."""

    # When additional columns are added to QGeometry, this is the example to populate it.
    # e.g. element_extensions = dict(
    #         base=dict(color=str, klayer=int),
    #         path=dict(thickness=float, material=str, perfectE=bool),
    #         poly=dict(thickness=float, material=str), )

    # Add columns to junction table during QGDSRenderer.load()
    # element_extensions  is now being populated as part of load().
    # Determined from element_table_data.

    # Dict structure MUST be same as  element_extensions!!!!!!
    # This dict will be used to update QDesign during init of renderer.
    # Keeping this as a cls dict so could be edited before renderer is instantiated.
    # To update component.options junction table.

    element_table_data = dict(
        # Example of adding a column named "empro_column_name"
        # with default values of "a_default_value" to the junction table.
        # Note: QEMProRenderer.name is prefixed to "a_column_name" when the table is appended by QComponents.
        junction=dict(a_column_name='a_default_value'))
    """element extensions dictionary   element_extensions = dict() from base class"""

    def __init__(self,
                 design: 'QDesign',
                 initiate=True,
                 render_template: Dict = None,
                 render_options: Dict = None):
        """Create a QRenderer for EMPro applications
        Args:
            design (QDesign): Use QGeometry within QDesign  to obtain elements.
            initiate (bool, optional): True to initiate the renderer. Defaults to True.
            render_template (Dict, optional): Typically used by GUI for template options for GDS.  Defaults to None.
            render_options (Dict, optional):  Used to overide all options. Defaults to None.
        """

        super().__init__(design=design,
                         initiate=initiate,
                         render_template=render_template,
                         render_options=render_options)
        QEMProRenderer.load()

        # Updated each time write_qgeometry_table_names_to_file() is called.
        self.chip_info = dict()

    # For a empro_renderer user, this is kept to exemplify self.logger.warning.

    def _initiate_renderer(self):
        """Not used by the empro renderer at this time. only returns True.
        """
        return True

    def _close_renderer(self):
        """Not used by the empro renderer at this time. only returns True.
        """
        return True

    def render_design(self):
        """Export the design to EMPro."""
        self.write_qgeometry_table_names_to_file(
            file_name=self.options.file_geometry_tables,
            highlight_qcomponents=[])

    def _can_write_to_path(self, file: str) -> int:
        """Check if can write file.

        Args:
            file (str): Has the path and/or just the file name.

        Returns:
            int: 1 if access is allowed. Else returns 0, if access not given.
        """
        status, directory_name = can_write_to_path(file)
        if status:
            return 1

        self.logger.warning(f'Not able to write to directory.'
                            f'File:"{file}" not written.'
                            f' Checked directory:"{directory_name}".')
        return 0

    def render_tables(self, skip_junction: bool = False):
        """
        Render components in design grouped by table type (path, poly, or junction).
        """
        for table_type in self.design.qgeometry.get_element_types():
            if table_type != 'junction' or not skip_junction:
                self.render_components(table_type)
                
    def render_components(self, table_type: str):
        """
        Render components by breaking them down into individual elements.

        Args:
            table_type (str): Table type (poly, path, or junction).
        """
        table = self.design.qgeometry.tables[table_type]

        if self.case == 0:  # Render a subset of components using mask
            mask = table['component'].isin(self.qcomp_ids)
            table = table[mask]

        for _, qgeom in table.iterrows():
            self.render_element(qgeom, bool(table_type == 'junction'))

        if table_type == 'path':
            self.auto_wirebonds(table)
            
    def render_element(self, qgeom: pd.Series, is_junction: bool):
        """Render an individual shape whose properties are listed in a row of
        QGeometry table. Junction elements are handled separately from non-
        junction elements, as the former consist of two rendered shapes, not
        just one.

        Args:
            qgeom (pd.Series): GeoSeries of element properties.
            is_junction (bool): Whether or not qgeom belongs to junction table.
        """
        qc_shapely = qgeom.geometry
        if is_junction:
            self.render_element_junction(qgeom)
        else:
            if isinstance(qc_shapely, shapely.geometry.Polygon):
                self.render_element_poly(qgeom)
            elif isinstance(qc_shapely, shapely.geometry.LineString):
                self.render_element_path(qgeom)
                
    def render_element_junction(self, qgeom: pd.Series):
        """
        Render a Josephson junction consisting of
            1. A rectangle of length pad_gap and width inductor_width. Defines lumped element
               RLC boundary condition.
            2. A line that is later used to calculate the voltage in post-processing analysis.

        Args:
            qgeom (pd.Series): GeoSeries of element properties.
        """
        ansys_options = dict(transparency=0.0)

        qc_name = 'Lj_' + str(qgeom['component'])
        qc_elt = get_clean_name(qgeom['name'])
        qc_shapely = qgeom.geometry
        qc_chip_z = parse_units(self.design.get_chip_z(qgeom.chip))
        qc_width = parse_units(qgeom.width)

        name = f'{qc_name}{QAnsysRenderer.NAME_DELIM}{qc_elt}'

        endpoints = parse_units(list(qc_shapely.coords))
        endpoints_3d = to_vec3D(endpoints, qc_chip_z)
        x0, y0, z0 = endpoints_3d[0]
        x1, y1, z0 = endpoints_3d[1]
        if abs(y1 - y0) > abs(x1 - x0):
            # Junction runs vertically up/down
            x_min, x_max = x0 - qc_width / 2, x0 + qc_width / 2
            y_min, y_max = min(y0, y1), max(y0, y1)
        else:
            # Junction runs horizontally left/right
            x_min, x_max = min(x0, x1), max(x0, x1)
            y_min, y_max = y0 - qc_width / 2, y0 + qc_width / 2

        # Draw rectangle
        self.logger.debug(f'Drawing a rectangle: {name}')
        poly_ansys = self.modeler.draw_rect_corner([x_min, y_min, qc_chip_z],
                                                   x_max - x_min, y_max - y_min,
                                                   qc_chip_z, **ansys_options)
        axis = 'x' if abs(x1 - x0) > abs(y1 - y0) else 'y'
        self.modeler.rename_obj(poly_ansys, 'JJ_rect_' + name)
        self.assign_mesh.append('JJ_rect_' + name)

        # Draw line
        poly_jj = self.modeler.draw_polyline([endpoints_3d[0], endpoints_3d[1]],
                                             closed=False,
                                             **dict(color=(128, 0, 128)))
        poly_jj = poly_jj.rename('JJ_' + name + '_')
        poly_jj.show_direction = True        
            
    def render_element_poly(self, qgeom: pd.Series):
        """Render a closed polygon.

        Args:
            qgeom (pd.Series): GeoSeries of element properties.
        """
        ansys_options = dict(transparency=0.0)

        qc_name = self.design._components[qgeom['component']].name
        qc_elt = get_clean_name(qgeom['name'])

        qc_shapely = qgeom.geometry  # shapely geom
        qc_chip_z = parse_units(self.design.get_chip_z(qgeom.chip))
        qc_fillet = round(qgeom.fillet, 7)

        name = f'{qc_elt}{QAnsysRenderer.NAME_DELIM}{qc_name}'

        points = parse_units(list(
            qc_shapely.exterior.coords))  # list of 2d point tuples
        points_3d = to_vec3D(points, qc_chip_z)

        if is_rectangle(qc_shapely):  # Draw as rectangle
            self.logger.debug(f'Drawing a rectangle: {name}')
            x_min, y_min, x_max, y_max = qc_shapely.bounds
            poly_ansys = self.modeler.draw_rect_corner(
                *parse_units([[x_min, y_min, qc_chip_z], x_max - x_min,
                              y_max - y_min, qc_chip_z]), **ansys_options)
            self.modeler.rename_obj(poly_ansys, name)

        else:
            # Draw general closed poly
            poly_ansys = self.modeler.draw_polyline(points_3d[:-1],
                                                    closed=True,
                                                    **ansys_options)
            # rename: handle bug if the name of the cut already exits and is used to make a cut
            poly_ansys = poly_ansys.rename(name)

        qc_fillet = round(qgeom.fillet, 7)
        if qc_fillet > 0:
            qc_fillet = parse_units(qc_fillet)
            idxs_to_fillet = good_fillet_idxs(
                points,
                qc_fillet,
                precision=self.design._template_options.PRECISION,
                isclosed=True)
            if idxs_to_fillet:
                self.modeler._fillet(qc_fillet, idxs_to_fillet, poly_ansys)

        # Subtract interior shapes, if any
        if len(qc_shapely.interiors) > 0:
            for i, x in enumerate(qc_shapely.interiors):
                interior_points_3d = to_vec3D(parse_units(list(x.coords)),
                                              qc_chip_z)
                inner_shape = self.modeler.draw_polyline(
                    interior_points_3d[:-1], closed=True)
                self.modeler.subtract(name, [inner_shape])

        # Input chip info into self.chip_subtract_dict
        if qgeom.chip not in self.chip_subtract_dict:
            self.chip_subtract_dict[qgeom.chip] = set()

        if qgeom['subtract']:
            self.chip_subtract_dict[qgeom.chip].add(name)

        # Potentially add to list of elements to metallize
        elif not qgeom['helper']:
            self.assign_perfE.append(name)
            
    def render_element_path(self, qgeom: pd.Series):
        """Render a path-type element.

        Args:
            qgeom (pd.Series): GeoSeries of element properties.
        """
        ansys_options = dict(transparency=0.0)

        qc_name = self.design._components[qgeom['component']].name
        qc_elt = get_clean_name(qgeom['name'])

        qc_shapely = qgeom.geometry  # shapely geom
        qc_chip_z = parse_units(self.design.get_chip_z(qgeom.chip))

        name = f'{qc_elt}{QAnsysRenderer.NAME_DELIM}{qc_name}'

        qc_width = parse_units(qgeom.width)

        points = parse_units(list(qc_shapely.coords))
        points_3d = to_vec3D(points, qc_chip_z)

        try:
            poly_ansys = self.modeler.draw_polyline(points_3d,
                                                    closed=False,
                                                    **ansys_options)
        except AttributeError:
            if self.modeler is None:
                self.logger.error(
                    'No modeler was found. Are you connected to an active Ansys Design?'
                )
            raise

        poly_ansys = poly_ansys.rename(name)

        qc_fillet = round(qgeom.fillet, 7)
        if qc_fillet > 0:
            qc_fillet = parse_units(qc_fillet)
            idxs_to_fillet = good_fillet_idxs(
                points,
                qc_fillet,
                precision=self.design._template_options.PRECISION,
                isclosed=False)
            if idxs_to_fillet:
                self.modeler._fillet(qc_fillet, idxs_to_fillet, poly_ansys)

        if qc_width:
            x0, y0 = points[0]
            x1, y1 = points[1]
            vlen = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
            p0 = np.array([
                x0, y0, 0
            ]) + qc_width / (2 * vlen) * np.array([y0 - y1, x1 - x0, 0])
            p1 = np.array([
                x0, y0, 0
            ]) + qc_width / (2 * vlen) * np.array([y1 - y0, x0 - x1, 0])
            shortline = self.modeler.draw_polyline([p0, p1],
                                                   closed=False)  # sweepline
            import pythoncom
            try:
                self.modeler._sweep_along_path(shortline, poly_ansys)
            except pythoncom.com_error as error:
                print("com_error: ", error)
                hr, msg, exc, arg = error.args
                if msg == "Exception occurred." and hr == -2147352567:
                    self.logger.error(
                        "We cannot find a writable design. \n  Either you are trying to use a Ansys "
                        "design that is not empty, in which case please clear it manually or with the "
                        "renderer method clean_active_design(). \n  Or you accidentally deleted "
                        "the design in Ansys, in which case please create a new one."
                    )
                raise error

        if qgeom.chip not in self.chip_subtract_dict:
            self.chip_subtract_dict[qgeom.chip] = set()

        if qgeom['subtract']:
            self.chip_subtract_dict[qgeom.chip].add(name)

        elif qgeom['width'] and (not qgeom['helper']):
            self.assign_perfE.append(name)

    def render_final_components(self,
                          file_name: str,
                          highlight_qcomponents: list = []) -> int:
        """Write out a file that can be used as an input by EMPro.
        The names will be for qcomponents that were selected or all of
        the qcomponents within the qdesign.

        Args:
            file_name (str): File name which can also include directory path.
                             If the file exists, it will be overwritten.
            highlight_qcomponents (list): List of strings which denote the name of QComponents to render.
                                        If empty, render all qcomponents in qdesign.

        Returns:
            int: 0=components can not be written to file, otherwise 1=components have been written
        """

        if not self._can_write_to_path(file_name):
            return 0

        self.chip_info.clear()
        
        status, table_names_used = self.get_qgeometry_tables_for_empro(highlight_qcomponents)
        
        intialization_text = ['#Clear project space and start fresh'  + '\n',
                              'import empro' + '\n',
                              'empro.activeProject.clear()' + '\n',
                              'from empro.toolkit.via_designer.geometry import makePolygon, makePolyLine' + '\n'+ '\n',
                              'width = 0.20' + '\n',
                              'height = 0.10' + '\n']
        
        #Iterate through tables and render components
        tables = self.get_table('','','')
            

        if (status == 0):
            empro_out = open(file_name, 'w')
            empro_out.writelines(intialization_text)
            empro_out.writelines(tables)
            empro_out.close()
            return 1
        else:
            return 0
        