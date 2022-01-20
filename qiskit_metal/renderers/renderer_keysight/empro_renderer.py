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

from shapely.geometry import LineString, mapping
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
    
    def render_poly(self):
        """This function finds all polygons in the current design and renders them to EMPro python script"""
        
        table = self.design.qgeometry.tables['poly']  
        
        coords = []
        for index, row in table.iterrows():
            coords_new = list(row['geometry'].exterior.coords)
            coords.append(coords_new)

        # Remove last tuple from each list
        rect = [] 
        for item in coords:
            rect_new = item[0:-1] 
            rect.append(rect_new) 
            
            
        # Add zero to each tuple to make it compatible with the EMPro format
        rect_mod = []
        for i in rect:
            for j in i:
                rect_mod_new = j + (0.0,)
                rect_mod.append(rect_mod_new)
                
        # Get the original number of elements in the geometry then split the list 
        x = index
        j = 0
        vertices = [0]*x
        for i in range(x):
            vertices[i] = rect_mod[j:j+4]
            j += 4

        return vertices

    def render_final_components(self,
                          file_name: str) -> int:
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
        vertices = self.render_poly()
        
        intialization_text = ['#Clear project space and start fresh'  + '\n',
                              'import empro' + '\n',
                              'empro.activeProject.clear()' + '\n',
                              'from empro.geometry import Sketch' + '\n',
                              'from empro.toolkit.geometry import sheetBody' + '\n',
                              'from empro.toolkit.via_designer.geometry import makePolygon, makePolyLine' + '\n'+ '\n',
                              "def makeSheetBody(listOfPolygonsByVertices, name=''):" + '\n',
                              '\t' + 'sketch = Sketch()' + '\n',
                              '\t' + 'for polygonVertices in listOfPolygonsByVertices:' + '\n',
                              '\t' + '\t' + 'makePolygon(polygonVertices,sketch)' + '\n',
                              '\t' + 'return sheetBody(sketch,name)' + '\n'+ '\n']
        
        add_geometries_text = [f'empro.activeProject.geometry.append(makeSheetBody({vertices},"Polygons"))']
        
        #Iterate through tables and render components
            

        
        empro_out = open(file_name, 'w')
        empro_out.writelines(intialization_text)
        empro_out.writelines(add_geometries_text)
        empro_out.close()

        