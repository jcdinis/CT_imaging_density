#http://en.wikipedia.org/wiki/Local_maximum_intensity_projection
import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport floor, ceil
from cython.parallel import prange

DTYPE = np.uint8
ctypedef np.uint8_t DTYPE_t

DTYPE16 = np.int16
ctypedef np.int16_t DTYPE16_t

@cython.boundscheck(False) # turn of bounds-checking for entire function
def lmip(np.ndarray[DTYPE16_t, ndim=3] image, int axis, DTYPE16_t tmin,
         DTYPE16_t tmax, np.ndarray[DTYPE16_t, ndim=2] out):
    cdef DTYPE16_t max
    cdef int start
    cdef int sz = image.shape[0]
    cdef int sy = image.shape[1]
    cdef int sx = image.shape[2]

    # AXIAL
    if axis == 0:
        for x in xrange(sx):
            for y in xrange(sy):
                max = image[0, y, x]
                if max >= tmin and max <= tmax:
                    start = 1
                else:
                    start = 0
                for z in xrange(sz):
                    if image[z, y, x] > max:
                        max = image[z, y, x]

                    elif image[z, y, x] < max and start:
                        break
                    
                    if image[z, y, x] >= tmin and image[z, y, x] <= tmax:
                        start = 1

                out[y, x] = max

    #CORONAL
    elif axis == 1:
        for z in xrange(sz):
            for x in xrange(sx):
                max = image[z, 0, x]
                if max >= tmin and max <= tmax:
                    start = 1
                else:
                    start = 0
                for y in xrange(sy):
                    if image[z, y, x] > max:
                        max = image[z, y, x]

                    elif image[z, y, x] < max and start:
                        break
                    
                    if image[z, y, x] >= tmin and image[z, y, x] <= tmax:
                        start = 1

                out[z, x] = max

    #CORONAL
    elif axis == 2:
        for z in xrange(sz):
            for y in xrange(sy):
                max = image[z, y, 0]
                if max >= tmin and max <= tmax:
                    start = 1
                else:
                    start = 0
                for x in xrange(sx):
                    if image[z, y, x] > max:
                        max = image[z, y, x]

                    elif image[z, y, x] < max and start:
                        break
                    
                    if image[z, y, x] >= tmin and image[z, y, x] <= tmax:
                        start = 1

                out[z, y] = max


cdef DTYPE16_t get_colour(DTYPE16_t vl, DTYPE16_t wl, DTYPE16_t ww):
    cdef DTYPE16_t out_colour
    cdef DTYPE16_t min_value = wl - (ww / 2)
    cdef DTYPE16_t max_value = wl + (ww / 2)
    if vl < min_value:
        out_colour = min_value
    elif vl > max_value:
        out_colour = max_value
    else:
        out_colour = vl

    return out_colour

@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.cdivision(True)
cdef float get_opacity(DTYPE16_t vl, DTYPE16_t wl, DTYPE16_t ww) nogil:
    cdef float out_opacity
    cdef DTYPE16_t min_value = wl - (ww / 2)
    cdef DTYPE16_t max_value = wl + (ww / 2)
    if vl < min_value:
        out_opacity = 0.0
    elif vl > max_value:
        out_opacity = 1.0
    else:
        out_opacity = 1.0/(max_value - min_value) * (vl - min_value)

    return out_opacity


@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.cdivision(True)
def mida(np.ndarray[DTYPE16_t, ndim=3] image, int axis, DTYPE16_t wl,
         DTYPE16_t ww, np.ndarray[DTYPE16_t, ndim=2] out):
    cdef int sz = image.shape[0]
    cdef int sy = image.shape[1]
    cdef int sx = image.shape[2]

    cdef DTYPE16_t min = image.min()
    cdef DTYPE16_t max = image.max()
    cdef DTYPE16_t vl

    cdef DTYPE16_t min_value = wl - (ww / 2)
    cdef DTYPE16_t max_value = wl + (ww / 2)

    cdef float fmax=0.0
    cdef float fpi
    cdef float dl
    cdef float bt

    cdef float alpha
    cdef float alpha_p = 0.0
    cdef float colour
    cdef float colour_p = 0

    cdef int x, y, z

    # AXIAL
    if axis == 0:
        for x in prange(sx, nogil=True):
            for y in xrange(sy):
                fmax = 0.0
                alpha_p = 0.0
                colour_p = 0.0
                for z in xrange(sz):
                    vl = image[z, y, x]
                    fpi = 1.0/(max - min) * (vl - min)
                    if fpi > fmax:
                        dl = fpi - fmax
                        fmax = fpi
                    else:
                        dl = 0.0

                    bt = 1.0 - dl
                    
                    colour = fpi
                    alpha = get_opacity(vl, wl, ww)
                    colour = (bt * colour_p) + (1 - bt * alpha_p) * colour * alpha
                    alpha = (bt * alpha_p) + (1 - bt * alpha_p) * alpha

                    colour_p = colour
                    alpha_p = alpha

                    if alpha >= 1.0:
                        break


                #out[y, x] = <DTYPE16_t>((max_value - min_value) * colour + min_value)
                out[y, x] = <DTYPE16_t>((max - min) * colour + min)


    #CORONAL
    elif axis == 1:
        for z in prange(sz, nogil=True):
            for x in xrange(sx):
                fmax = 0.0
                alpha_p = 0.0
                colour_p = 0.0
                for y in xrange(sy):
                    vl = image[z, y, x]
                    fpi = 1.0/(max - min) * (vl - min)
                    if fpi > fmax:
                        dl = fpi - fmax
                        fmax = fpi
                    else:
                        dl = 0.0

                    bt = 1.0 - dl
                    
                    colour = fpi
                    alpha = get_opacity(vl, wl, ww)
                    colour = (bt * colour_p) + (1 - bt * alpha_p) * colour * alpha
                    alpha = (bt * alpha_p) + (1 - bt * alpha_p) * alpha

                    colour_p = colour
                    alpha_p = alpha

                    if alpha >= 1.0:
                        break

                out[z, x] = <DTYPE16_t>((max - min) * colour + min)

    #AXIAL
    elif axis == 2:
        for z in prange(sz, nogil=True):
            for y in xrange(sy):
                fmax = 0.0
                alpha_p = 0.0
                colour_p = 0.0
                for x in xrange(sx):
                    vl = image[z, y, x]
                    fpi = 1.0/(max - min) * (vl - min)
                    if fpi > fmax:
                        dl = fpi - fmax
                        fmax = fpi
                    else:
                        dl = 0.0

                    bt = 1.0 - dl
                    
                    colour = fpi
                    alpha = get_opacity(vl, wl, ww)
                    colour = (bt * colour_p) + (1 - bt * alpha_p) * colour * alpha
                    alpha = (bt * alpha_p) + (1 - bt * alpha_p) * alpha

                    colour_p = colour
                    alpha_p = alpha

                    if alpha >= 1.0:
                        break

                out[z, y] = <DTYPE16_t>((max - min) * colour + min)
