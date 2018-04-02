import glob
import os
import tempfile
import sys

import numpy
import vtkgdcm
import gdcm
import vtk

from vtk.util import numpy_support

def dcm2mmap(input_dir, output_mmap):
    tmp_dcm_files = glob.glob(os.path.join(input_dir, '*'))

    sorter = gdcm.IPPSorter()
    sorter.Sort(tmp_dcm_files)
    dcm_files = sorter.GetFilenames()

    if not dcm_files:
        dcm_files = sorted(tmp_dcm_files)

    r = vtkgdcm.vtkGDCMImageReader()
    r.SetFileName(dcm_files[0])
    r.Update()

    x, y, z = r.GetOutput().GetDimensions()
    
    a_size = len(dcm_files), y, x
    t = numpy_support.get_numpy_array_type(r.GetDataScalarType())
    print t

    print t, a_size
    m_dcm = numpy.memmap(output_mmap, mode='w+', dtype='int16', shape=a_size)

    lp = 0
    zspacing = 0

    for i, dcm_file in enumerate(dcm_files):
        r = vtkgdcm.vtkGDCMImageReader()
        r.SetFileName(dcm_file)
        r.Update()

        if lp:
            zspacing += abs(r.GetImagePositionPatient()[2] - lp)
        lp = r.GetImagePositionPatient()[2]

        o = r.GetOutput()
        d = numpy_support.vtk_to_numpy(o.GetPointData().GetScalars())

        m_dcm[i] = d.reshape(y, x)

    m_dcm.flush()

    xs, ys, zs = o.GetSpacing()
    spacing = xs, ys, zspacing / (len(dcm_files) - 1.0)
    print spacing
    return m_dcm, spacing

def make_mask(image, mask_file):
    """
    Makes a mask from a image
    """
    m_shape = tuple([i + 1 for i in image.shape])
    mask =  numpy.memmap(mask_file, mode='w+', dtype='uint8', shape=m_shape)
    mask.flush()
    return mask

def main():
    dcm2mmap(sys.argv[1], sys.argv[2])

if __name__ == '__main__':
    main()
