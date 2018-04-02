# -*- coding: utf-8*-
import math
import multiprocessing
import plistlib
import sys
import tempfile

import constants as const
import wx
import numpy
from scipy import ndimage
import vtk
from vtk.wx.wxVTKRenderWindowInteractor import wxVTKRenderWindowInteractor
from vtk.util import numpy_support
from dcm2mmap import dcm2mmap, make_mask
from skimage.filters import threshold_otsu

from wx.lib.pubsub import setuparg1

import wx.lib.pubsub.pub as Publisher

from PIL import Image, ImageDraw

#import mahotas
import Density

#import floodfill
import clut_imagedata
import lmip

NUMPY_TO_VTK_TYPE = {
                     'int16': 'SetScalarTypeToShort',
                     'uint16': 'SetScalarTypeToUnsignedShort',
                     'uint8': 'SetScalarTypeToUnsignedChar',
                     'float32': 'SetScalarTypeToFloat'
                    }


wildcard = "VTK image file (*.vti)|*.vti|" \
            "All files (*.*)|*.*"

def histeq(im,nbr_bins=256):
    maximum = ndimage.maximum(im)
    minimum = ndimage.minimum(im)
    #get image histogram
    imhist,bins = numpy.histogram(im.flat,nbr_bins,normed=True)
    cdf = imhist.cumsum() #cumulative distribution function
    cdf = 255 * cdf / cdf[-1] #normalize

    #use linear interpolation of cdf to find new pixel values
    im2 = numpy.interp(im.flat,bins[:-1],cdf)

    print maximum, minimum
    return im2.reshape(im.shape), cdf


def to_vtk(n_array, spacing):
    dz, dy, dx = n_array.shape
    n_array.shape = dx * dy * dz

    v_image = numpy_support.numpy_to_vtk(n_array)

    # Generating the vtkImageData
    image = vtk.vtkImageData()
    image.SetDimensions(dx, dy, dz)
    image.SetOrigin(0, 0, 0)
    image.SetSpacing(spacing)
    image.SetExtent(0, dx -1, 0, dy -1, 0, dz - 1)

    image.AllocateScalars(numpy_support.get_vtk_array_type(n_array.dtype),1)
    image.GetPointData().SetScalars(v_image)

    image_copy = vtk.vtkImageData()
    image_copy.DeepCopy(image)

    n_array.shape = dz, dy, dx
    return image_copy

# widget que com um slider e um campo de texto
class SliderText(wx.Panel):
    def __init__(self, parent, id, value, Min, Max):
        wx.Panel.__init__(self, parent, id)
        self.min = Min
        self.max = Max
        self.value = value
        self.build_gui()
        self.__bind_events_wx()
        self.Show()

    def build_gui(self):
        self.sliderctrl = wx.Slider(self, -1, self.value, self.min, self.max)
        self.textbox = wx.TextCtrl(self, -1, "%d" % self.value)


        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.sliderctrl, 1, wx.EXPAND)
        sizer.Add(self.textbox, 0, wx.EXPAND)
        self.SetSizer(sizer)

        self.Layout()
        self.Update()
        self.SetAutoLayout(1)

    def __bind_events_wx(self):
        self.sliderctrl.Bind(wx.EVT_SCROLL, self.do_slider)
        self.Bind(wx.EVT_SIZE, self.onsize)

    def onsize(self, evt):
        print "OnSize"
        evt.Skip()

    def do_slider(self, evt):
        self.value =  self.sliderctrl.GetValue()
        self.textbox.SetValue("%d" % self.value)
        evt.Skip()

    def GetValue(self):
        return self.value


class Window(wx.Frame):
    def __init__(self, parent):
        wx.Frame.__init__(self, parent, size=(500, 500))
        self.build_gui()
        #self.run_image()
        self.__bind_events()
        self.__bind_events_wx()
        self.Show()

    def build_gui(self):
        self.axial_viewer = Viewer(self, 'AXIAL')
        self.coronal_viewer = Viewer(self, 'CORONAL')
        self.sagital_viewer = Viewer(self, 'SAGITAL')
        self.cortical_viewer = Viewer(self, 'AXIAL', 'DENSIDADE', 'Osso cortical')
        self.trabecular_viewer = Viewer(self, 'AXIAL', 'DENSIDADE', 'Osso trabecular')


        self.surface_button = wx.Button(self, -1, "Surface")
        self.contorno_btn = wx.Button(self, -1, "Contorno")
        self.clut_btn = wx.Button(self, -1, "CLUT")
        self.calculo_density1=wx.Button(self,-1,"Calculo Density")
        self.calculo_modulo_young=wx.Button(self,-1, "Calculo Modulo de Young")
        self.retira_suporte=wx.Button(self,-1,'Retirar Suporte')



        self.threshold_min = SliderText(self, -1, 300, -1024, 3033)
        self.threshold_max = SliderText(self, -1, 3033, -1024, 3033)

        self.mip_size = SliderText(self, -1, 1, 1, 128)
        self.ww = SliderText(self, -1, 255, 0, 3033)
        self.wl = SliderText(self, -1, 127, -1024, 3033)


        self.mip_type = wx.ComboBox(self, -1, "Max", choices=("Max", "Min",
                                                              "Mean", "Median",
                                                              "LMIP", "MIDA"),


                                    style=wx.CB_READONLY)

        #escolher a funcao para calculo da densidade
        self.autores_densidade=wx.ComboBox(self, -1, "Taylor_Roland_2002", choices=("Taylor_Roland_2002",
                                                                            "Pedro_200"),
                                    style=wx.CB_READONLY)

        #escolher a funcao para calculo do modulo de elasticidade

        self.autores_Modulo_young=wx.ComboBox(self, -1, "Carter_Haynes_1977", choices=("Carter_Haynes_1977",
                                                                                        "Rice_et_al_1988","Rho_et_al_1995", "Morgan_et_al_2003",
                                                                                            "Keller_1994",
                                                                                                "Peng_et_al_2006_e_Wirtz_et_al_2000"),
                                    style=wx.CB_READONLY)


        self.sampleList = ['Direccao Z', 'Direccao Y', 'Direccao X']

        self.rb = wx.RadioBox(
                self, -1, "Direcao de plotagem", wx.DefaultPosition, wx.DefaultSize,
                self.sampleList, 2, wx.RA_SPECIFY_COLS
                )

        # regiao de corte


        self.z_inicial= wx.SpinCtrl(self, -1, "",(10,50))
        self.z_final= wx.SpinCtrl(self, -1, "",(100,30))


        self.y_inicial= wx.SpinCtrl(self, -1, "",(10,50))
        self.y_final= wx.SpinCtrl(self, -1, "",(100,30))

        self.x_inicial= wx.SpinCtrl(self, -1, "",(10,50))
        self.x_final= wx.SpinCtrl(self, -1, "",(100,30))



        viewer_sizer = wx.BoxSizer(wx.HORIZONTAL)
        viewer_sizer.Add(self.axial_viewer, 1, wx.EXPAND|wx.GROW)
        viewer_sizer.Add(self.sagital_viewer, 1, wx.EXPAND|wx.GROW)
        viewer_sizer.Add(self.coronal_viewer, 1, wx.EXPAND|wx.GROW)
        viewer_densidade_sizer = wx.BoxSizer(wx.HORIZONTAL)
        viewer_densidade_sizer.Add(self.cortical_viewer, 1, wx.EXPAND|wx.GROW)
        viewer_densidade_sizer.Add(self.trabecular_viewer, 1, wx.EXPAND|wx.GROW)

        b_sizer = wx.BoxSizer(wx.VERTICAL)

        viewers_sizer = wx.BoxSizer(wx.VERTICAL)
        viewers_sizer.Add(viewer_sizer, 10, wx.EXPAND)
        viewers_sizer.Add(viewer_densidade_sizer, 10, wx.EXPAND)

        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(viewers_sizer, 10, wx.EXPAND)
        sizer.Add(b_sizer, 3, wx.EXPAND)

        b_sizer.Add(self.mip_type)
        b_sizer.Add(self.surface_button)
        b_sizer.Add(self.contorno_btn)
        b_sizer.Add(self.retira_suporte)

        b_sizer.Add(self.clut_btn)

        b_sizer.Add(wx.StaticText(self, -1, "MiP Size") , 0, wx.EXPAND)
        b_sizer.Add(self.mip_size, 0, wx.EXPAND)

        b_sizer.Add(wx.StaticText(self, -1, "Threshold") , 0, wx.EXPAND)
        b_sizer.Add(self.threshold_min, 0, wx.EXPAND)
        b_sizer.Add(self.threshold_max, 0, wx.EXPAND)

        b_sizer.Add(wx.StaticText(self, -1, "WL & WL") , 0, wx.EXPAND)
        b_sizer.Add(self.wl, 0, wx.EXPAND)
        b_sizer.Add(self.ww, 0, wx.EXPAND)

        b_sizer.Add(wx.StaticText(self, -1, "Metodo Calculo Densidade") , 0, wx.EXPAND)
        b_sizer.Add(self.autores_densidade)

        b_sizer.Add(wx.StaticText(self, -1, "Processamento Densidade") , 0, wx.EXPAND)
        b_sizer.Add(self.calculo_density1)

        b_sizer.Add(wx.StaticText(self, -1, "Metodo Calculo Modulo de Young") , 0, wx.EXPAND)
        b_sizer.Add(self.autores_Modulo_young)

        b_sizer.Add(wx.StaticText(self, -1, "Processamento Modulo de Young") , 0, wx.EXPAND)
        b_sizer.Add(self.calculo_modulo_young)

        b_sizer.Add(wx.StaticText(self, -1, "Direccao de Visualizacao da Plotagem") , 0, wx.EXPAND)
        b_sizer.Add(self.rb,0)
        b_sizer.Add(wx.StaticText(self, -1, "posicionamente para corte") , 0, wx.EXPAND)

        z_sizer=wx.BoxSizer(wx.HORIZONTAL)
        z_sizer.Add(wx.StaticText(self, -1, "Direccao Z de") , 0, wx.EXPAND)
        z_sizer.Add(self.z_inicial,0,wx.EXPAND)
        z_sizer.Add(wx.StaticText(self, -1, "até") , 0, wx.EXPAND)
        z_sizer.Add(self.z_final,0,wx.EXPAND)

        y_sizer=wx.BoxSizer(wx.HORIZONTAL)
        y_sizer.Add(wx.StaticText(self, -1, "Direccao Y de") , 0, wx.EXPAND)
        y_sizer.Add(self.y_inicial,0,wx.EXPAND)
        y_sizer.Add(wx.StaticText(self, -1, "até") , 0, wx.EXPAND)
        y_sizer.Add(self.y_final,0,wx.EXPAND)

        x_sizer=wx.BoxSizer(wx.HORIZONTAL)
        x_sizer.Add(wx.StaticText(self, -1, "Direccao X de") , 0, wx.EXPAND)
        x_sizer.Add(self.x_inicial,0,wx.EXPAND)
        x_sizer.Add(wx.StaticText(self, -1, "até") , 0, wx.EXPAND)
        x_sizer.Add(self.x_final,0,wx.EXPAND)


        #button cut matriz
        self.Matrix_cut_button = wx.Button(self, -1, "Cut")


        b_sizer.Add(z_sizer, 0, wx.EXPAND)
        b_sizer.Add(y_sizer, 0, wx.EXPAND)
        b_sizer.Add(x_sizer, 0, wx.EXPAND)
        b_sizer.Add(self.Matrix_cut_button)

        self.SetSizer(sizer)

        ############################# MENU #######################
        MenuBar=wx.MenuBar()

        menubar = wx.MenuBar()
        fileMenu = wx.Menu()
        fitem = fileMenu.Append(wx.ID_EXIT, 'Quit', 'Quit application')
        mOpen=fileMenu.Append(-1, 'Abrir Dicom','Abrir Dicom' )
        msave = fileMenu.Append(-1, 'Save Mask', 'Save Mask')
        isave = fileMenu.Append(-1, 'Save Image', 'Save Image')
        menubar.Append(fileMenu, '&File')


        menuControl=wx.Menu()
        #sub menu
        Densidade_menu = wx.Menu()
        Young_Modulo_menu= wx.Menu()
        menu_cortical=wx.Menu()
        menu_trabecular=wx.Menu()

        self.SetMenuBar(menubar)

        self.Bind(wx.EVT_MENU, self.OnQuit, fitem)
        self.Bind(wx.EVT_MENU, self.OnSaveMask, msave)
        self.Bind(wx.EVT_MENU, self.OnSaveImage, isave)
        self.Bind(wx.EVT_MENU, self.mOpenImagem, mOpen)

        ##########################################################

        self.Layout()
        self.Update()
        self.SetAutoLayout(1)

    def mOpenImagem(self, evt):
        dialog = wx.DirDialog(None, "Choose a directory:",style=wx.DD_DEFAULT_STYLE | wx.DD_NEW_DIR_BUTTON)
        if dialog.ShowModal() == wx.ID_OK:
            receberImagems=dialog.GetPath().encode("latin1")
            print receberImagems
            self.run_image(receberImagems)
        dialog.Destroy()

    def run_image(self, input_dir):
        self.image_file = tempfile.mktemp()
        m_input, self.spacing = dcm2mmap(input_dir, self.image_file)
        self.mask_file = tempfile.mktemp()
        mask = make_mask(m_input, self.mask_file)
        mask[:] = 0
        mask[0, :, :] = 1
        mask[:, 0, :] = 1
        mask[:, :, 0] = 1
        self.axial_viewer.SetInput(m_input, mask, self.spacing)
        self.sagital_viewer.SetInput(m_input, mask, self.spacing)
        self.coronal_viewer.SetInput(m_input, mask, self.spacing)

        self.markers = numpy.zeros(m_input.shape, 'int8')

        self.m_input = m_input
        self.z_corte,self.y_corte,self.x_corte=self.m_input.shape
        print 'hhhhhhhhhhhhhhhhh'
        print self.z_corte,self.y_corte,self.x_corte




        self.z_inicial.SetRange(0, self.z_corte-1)
        self.z_inicial.SetValue(0)

        self.z_final.SetRange(1,self.z_corte-1)
        self.z_final.SetValue(self.z_corte-1)

        self.y_inicial.SetRange(0,self.y_corte-1)
        self.y_inicial.SetValue(0)

        self.y_final.SetRange(1,self.y_corte-1)
        self.y_final.SetValue(self.y_corte-1)


        self.x_inicial.SetRange(0,self.x_corte-1)
        self.x_inicial.SetValue(0)

        self.x_final.SetRange(1,self.x_corte-1)
        self.x_final.SetValue(self.x_corte-1)



        self.m_mask = mask

    def __bind_events(self):
        Publisher.subscribe(self.add_marker,
                                 'Add marker')

    def __bind_events_wx(self):
        self.surface_button.Bind(wx.EVT_BUTTON, self.do_surface)
        self.contorno_btn.Bind(wx.EVT_BUTTON, self.do_contorno)
        self.calculo_density1.Bind(wx.EVT_BUTTON,self.calculo_density)
        self.calculo_modulo_young.Bind(wx.EVT_BUTTON,self.calculos_modulos_young)
        self.Matrix_cut_button.Bind(wx.EVT_BUTTON,self.faz_corte)
        self.retira_suporte.Bind(wx.EVT_BUTTON, self.do_remove_support)


        self.mip_size.Bind(wx.EVT_SCROLL, self.set_mip_size)
        self.mip_type.Bind(wx.EVT_COMBOBOX, self.OnSelectMIPType)

        self.threshold_min.Bind(wx.EVT_SCROLL, self.do_threshold)
        self.threshold_min.Bind(wx.EVT_SCROLL_THUMBRELEASE,
                                self.OnReleaseThreshold)

        self.threshold_max.Bind(wx.EVT_SCROLL, self.do_threshold)
        self.threshold_max.Bind(wx.EVT_SCROLL_THUMBRELEASE,
                                self.OnReleaseThreshold)

        self.ww.Bind(wx.EVT_SCROLL, self.do_ww_wl)
        self.wl.Bind(wx.EVT_SCROLL, self.do_ww_wl)

        self.clut_btn.Bind(wx.EVT_BUTTON, self.do_clut)

        self.rb.Bind(wx.EVT_RADIOBOX, self.Plotar_Direccao_plotagem)

        self.z_inicial.Bind(wx.EVT_SPINCTRL, self.cut_calculate)
        self.z_final.Bind(wx.EVT_SPINCTRL, self.cut_calculate)
        self.y_inicial.Bind(wx.EVT_SPINCTRL, self.cut_calculate)
        self.y_final.Bind(wx.EVT_SPINCTRL, self.cut_calculate)
        self.x_inicial.Bind(wx.EVT_SPINCTRL, self.cut_calculate)
        self.x_final.Bind(wx.EVT_SPINCTRL, self.cut_calculate)



    def cut_calculate(self,evt):
        zi=self.z_inicial.GetValue()
        zf=self.z_final.GetValue()
        yi=self.y_inicial.GetValue()
        yf=self.y_final.GetValue()
        xi=self.x_inicial.GetValue()
        xf=self.x_final.GetValue()

        self.axial_viewer.desenha_linhas(xi,xf,yi,yf,zi,zf)
        self.coronal_viewer.desenha_linhas(xi,xf,yi,yf,zi,zf)
        self.sagital_viewer.desenha_linhas(xi,xf,yi,yf,zi,zf)






    def Plotar_Direccao_plotagem(self, evt):
        if evt.GetInt()==0:
            orientation = 'AXIAL'
        elif evt.GetInt()==1:
            orientation = 'CORONAL'
        elif evt.GetInt()==2:
            orientation = 'SAGITAL'

        self.cortical_viewer.SetOrientation(orientation)
        self.trabecular_viewer.SetOrientation(orientation)

        Publisher.sendMessage('Update render', None)

    #funcao corte
    def faz_corte(self, evt):
        zi=self.z_inicial.GetValue()
        zf=self.z_final.GetValue()
        yi=self.y_inicial.GetValue()
        yf=self.y_final.GetValue()
        xi=self.x_inicial.GetValue()
        xf=self.x_final.GetValue()

        self.m_input = self.m_input[zi:zf,yi:yf,xi:xf]

        self.mask_file = tempfile.mktemp()
        mask = make_mask(self.m_input, self.mask_file)
        mask[:] = 0
        mask[0, :, :] = 1
        mask[:, 0, :] = 1
        mask[:, :, 0] = 1

        self.m_mask = mask

        self.axial_viewer.SetInput(self.m_input, mask, self.spacing)
        self.sagital_viewer.SetInput(self.m_input, mask, self.spacing)
        self.coronal_viewer.SetInput(self.m_input, mask, self.spacing)

        self.z_corte,self.y_corte,self.x_corte=self.m_input.shape
        print 'hhhhhhhhhhhhhhhhh'
        print self.z_corte,self.y_corte,self.x_corte




        self.z_inicial.SetRange(0, self.z_corte-1)
        self.z_inicial.SetValue(0)

        self.z_final.SetRange(1,self.z_corte-1)
        self.z_final.SetValue(self.z_corte-1)

        self.y_inicial.SetRange(0,self.y_corte-1)
        self.y_inicial.SetValue(0)

        self.y_final.SetRange(1,self.y_corte-1)
        self.y_final.SetValue(self.y_corte-1)


        self.x_inicial.SetRange(0,self.x_corte-1)
        self.x_inicial.SetValue(0)

        self.x_final.SetRange(1,self.x_corte-1)
        self.x_final.SetValue(self.x_corte-1)

        zi=self.z_inicial.GetValue()
        zf=self.z_final.GetValue()
        yi=self.y_inicial.GetValue()
        yf=self.y_final.GetValue()
        xi=self.x_inicial.GetValue()
        xf=self.x_final.GetValue()

        self.axial_viewer.desenha_linhas(xi,xf,yi,yf,zi,zf)
        self.coronal_viewer.desenha_linhas(xi,xf,yi,yf,zi,zf)
        self.sagital_viewer.desenha_linhas(xi,xf,yi,yf,zi,zf)


    def do_remove_support(self, evt):
        minimum = ndimage.minimum(self.m_input)
        maximum = ndimage.maximum(self.m_input)
        temp_file = tempfile.mktemp()
        temp_matrix = numpy.memmap(temp_file, mode='w+', dtype='uint16',
                                   shape=self.m_input.shape)
        temp_matrix[:] = self.m_input + abs(minimum)
        temp_matrix[:] = histeq(temp_matrix)[0]

        print minimum
        print temp_matrix.dtype
        T = threshold_otsu(temp_matrix)
        print ">>> T", T

        temp_matrix[temp_matrix < T] = 0
        #temp_matrix[temp_matrix >= T] = 255
        l, nr = ndimage.label(temp_matrix)
        ls = l.shape
        l = l.flatten()
        t = [(len(l[l==i]), i) for i in xrange(1, nr)]
        l.shape = ls
        print nr, t
        m = max(t)
        print m, nr
        self.m_input[l != m[1]] = minimum
        #self.m_input[:] = temp_matrix - minimum



    def set_mip_size(self, evt):
        #print "MIP size"
        Publisher.sendMessage('Set mip size', self.mip_size.GetValue())

    def OnSelectMIPType(self, evt):
        #print "On Select MIP Type"
        Publisher.sendMessage('Set mip type', self.mip_type.GetValue())


    def do_ww_wl(self, evt):
        print "ww wl"
        Publisher.sendMessage('Set wl ww',
                                   (self.wl.GetValue(),
                                    self.ww.GetValue()))

    def do_clut(self, evt):
        i, e = self.m_input.min(), self.m_input.max()
        r = e - i
        h, l = numpy.histogram(self.m_input, r, (i, e))
        cw = clut_imagedata.ClutDialog(self, -1, h, i, e,
                                                self.wl.GetValue(),
                                                self.ww.GetValue())
        cw.Bind(clut_imagedata.EVT_CLUT_POINT_MOVE, self.OnClutChange)
        cw.Show()



    def OnClutChange(self, evt):
        print evt.GetNodes()
        Publisher.sendMessage('Set nodes', evt.GetNodes())

    def OnCheckClog(self, evt):
        print "Check", self.check_clog.Value
        self.axial_viewer.islog = self.check_clog.Value
        self.sagital_viewer.islog = self.check_clog.Value
        self.coronal_viewer.islog = self.check_clog.Value

        Publisher.sendMessage('Update render', None)

    def add_marker(self, pubsub_evt):
        position, value = pubsub_evt.data
        self.markers[position] = value
        print "marker added", position, value


    def do_contorno(self, evt):
        self.m_input = ndimage.sobel(self.m_input)
        self.mask = make_mask(self.m_input, self.mask_file)
        self.mask[0, :, :] = 1
        self.mask[:, 0, :] = 1
        self.mask[:, :, 0] = 1
        self.axial_viewer.SetInput(self.m_input, self.mask, self.spacing)

    def generate_surface(self, image):
        t_min, t_max = (self.threshold_min.GetValue(),
                        self.threshold_max.GetValue())
        mcubes = vtk.vtkContourFilter()
        mcubes.GenerateValues(3, t_min, t_max)
        mcubes.SetInputData(image)
        mcubes.UseScalarTreeOn()
        mcubes.ComputeGradientsOn()
        mcubes.ComputeNormalsOn()
        mcubes.ComputeScalarsOn()
        mcubes.Update()


        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputData(mcubes.GetOutput())
        smoother.SetNumberOfIterations(2)
        smoother.SetFeatureAngle(30)
        smoother.FeatureEdgeSmoothingOn()
        smoother.BoundarySmoothingOn()
        smoother.Update()
        smoother.GetOutput().ReleaseDataFlagOn()

        n_input = smoother

        print "Normals"
        normals = vtk.vtkPolyDataNormals()
        # normals.GetOutput().ReleaseDataFlagOff()
        normals.SetInputData(n_input.GetOutput())
        normals.SetFeatureAngle(80)
        normals.Update()

        print "Stripper"
        stripper = vtk.vtkStripper()
        stripper.SetInputData(normals.GetOutput())
        # stripper.GetOutput().ReleaseDataFlagOff()
        stripper.Update()

        #output = tempfile.mktemp(prefix=str(self.ident), suffix='.vtp', dir=self.output_dir)

        stlBinary = vtk.vtkXMLPolyDataWriter()
        stlBinary.SetInputData(smoother.GetOutput())
        stlBinary.SetFileName('/tmp/superficie_do_camboja.vtp')
        stlBinary.Write()


    def do_surface(self, evt):
        image = to_vtk(self.m_input, self.spacing)
        output = self.generate_surface(image)

    def do_threshold(self, evt):
        Publisher.sendMessage('Set threshold',
                                   (self.threshold_min.GetValue(),
                                    self.threshold_max.GetValue()))
        Publisher.sendMessage('Update render', None)

    def OnReleaseThreshold(self, evt):
        print "OnRelease"
        t_min = self.threshold_min.GetValue()
        t_max = self.threshold_max.GetValue()

        for n, slice_ in enumerate(self.m_input):
            m = numpy.ones(slice_.shape, self.m_mask.dtype)
            m[slice_ < t_min] = 0
            m[slice_ > t_max] = 0
            m[m == 1] = 255
            self.m_mask[n+1, 1:, 1:] = m #.astype('uint8')

        self.m_mask[0,:,:] = 1
        self.m_mask[:,0,:] = 1
        self.m_mask[:,:,0] = 1
        Publisher.sendMessage('Update render', None)

    def OnQuit(self, e):
        self.Destroy()

    def OnSaveMask(self, e):
        path = self._dialog_save()
        if path:
            mask = numpy.array(self.m_mask[1:, 1:, 1:])
            vtkimg = to_vtk(mask, self.spacing)

            w = vtk.vtkXMLImageDataWriter()
            w.SetFileName(path)
            w.SetInputData(vtkimg)
            w.Write()

    def OnSaveImage(self, e):
        path = self._dialog_save()
        if path:
            vtkimg = to_vtk(self.m_input, self.spacing)

            w = vtk.vtkXMLImageDataWriter()
            w.SetFileName(path)
            w.SetInputData(vtkimg)
            w.Write()

    def _dialog_save(self):
        """
        Create and show the Save FileDialog
        """
        dlg = wx.FileDialog(
            self, message="Save file as ...",
            defaultDir='',
            defaultFile="", wildcard=wildcard, style=wx.SAVE
            )
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
        else:
            path = ''
        dlg.Destroy()

        return path

    def calculo_density(self,evt):

        if self.autores_densidade.GetValue()=="Taylor_Roland_2002":
            self.density_cortical, self.density_cancellous = Density.Taylor_W_R_Roland(self.m_input,self.spacing[0],self.spacing[1],self.spacing[2])

            numpy.save('densidade_cortical.npy', self.density_cortical)
            numpy.save('densidade_cancellous.npy', self.density_cancellous)

            mask =  numpy.zeros_like(self.density_cortical)
            self.cortical_viewer.SetInput(self.density_cortical, mask, self.spacing, 'DENSIDADE')
            #self.cortical_viewer.tipo = 'DENSIDADE'
            self.trabecular_viewer.SetInput(self.density_cancellous, mask, self.spacing, 'DENSIDADE')
            #self.trabecular_viewer.tipo = 'DENSIDADE'



        elif self.autores_densidade.GetValue()=="Pedro_200":
            self.density_cortical, self.density_cancellous = Density.densidade_Pedro_2000(self.m_input,self.spacing[0],self.spacing[1],self.spacing[2])
            numpy.save('densidade_cortical.npy', self.density_cortical)
            numpy.save('densidade_cancellous.npy', self.density_cancellous)

            mask =  numpy.zeros_like(self.density_cortical)
            self.cortical_viewer.SetInput(self.density_cortical, mask, self.spacing, 'DENSIDADE')
            self.trabecular_viewer.SetInput(self.density_cancellous, mask, self.spacing, 'DENSIDADE')



    def calculos_modulos_young(self,evt):

        if self.autores_Modulo_young.GetValue()=="Carter_Haynes_1977":
            matrix_modulo_young_cortical=Density.MY_Carter_Haynes_1977(self.density_cortical,self.spacing[0],self.spacing[1],self.spacing[2])
            numpy.save('modulo_young_cortical.npy', matrix_modulo_young_cortical)
            mask =  numpy.zeros_like(matrix_modulo_young_cortical)
            self.cortical_viewer.SetInput(matrix_modulo_young_cortical, mask, self.spacing, 'ELASTICIDADE')


        elif self.autores_Modulo_young.GetValue()=="Rice_et_al_1988":
            matrix_modulo_young_trabecular=Density.MY_Rice_et_al_1988(self.density_cancellous,self.spacing[0],self.spacing[1],self.spacing[2])
            numpy.save('modulo_young_trabecular.npy', matrix_modulo_young_trabecular)
            mask =  numpy.zeros_like(matrix_modulo_young_trabecular)
            self.trabecular_viewer.SetInput(matrix_modulo_young_trabecular, mask, self.spacing, 'ELASTICIDADE')



        elif self.autores_Modulo_young.GetValue()=="Rho_et_al_1995":
            matrix_modulo_young_cortical,matrix_modulo_young_trabecular=Density.MY_Rho_et_al_1995(self.density_cortical,self.density_cancellous,self.spacing[0],self.spacing[1],self.spacing[2])

            numpy.save('modulo_young_cortical.npy', matrix_modulo_young_cortical)
            numpy.save('modulo_young_trabecular.npy', matrix_modulo_young_trabecular)
            mask =  numpy.zeros_like(matrix_modulo_young_cortical)
            self.cortical_viewer.SetInput(matrix_modulo_young_cortical, mask, self.spacing, 'ELASTICIDADE')
            self.trabecular_viewer.SetInput(matrix_modulo_young_trabecular, mask, self.spacing, 'ELASTICIDADE')


        elif self.autores_Modulo_young.GetValue()=="Morgan_et_al_2003":
            matrix_modulo_young_cortical,matrix_modulo_young_trabecular=Density.MY_Morgan_et_al_2003(self.density_cortical,self.density_cancellous,self.spacing[0],self.spacing[1],self.spacing[2])

            numpy.save('modulo_young_cortical.npy', matrix_modulo_young_cortical)
            numpy.save('modulo_young_trabecular.npy', matrix_modulo_young_trabecular)
            mask =  numpy.zeros_like(matrix_modulo_young_cortical)
            self.cortical_viewer.SetInput(matrix_modulo_young_cortical, mask, self.spacing, 'ELASTICIDADE')
            self.trabecular_viewer.SetInput(matrix_modulo_young_trabecular, mask, self.spacing, 'ELASTICIDADE')


        elif self.autores_Modulo_young.GetValue()=="Keller_1994":
            matrix_modulo_young_cortical=Density.MY_Keller_1994(self.density_cortical,self.spacing[0],self.spacing[1],self.spacing[2])

            numpy.save('modulo_young_cortical.npy', matrix_modulo_young_cortical)
            mask =  numpy.zeros_like(matrix_modulo_young_cortical)
            self.cortical_viewer.SetInput(matrix_modulo_young_cortical, mask, self.spacing, 'ELASTICIDADE')


        elif self.autores_Modulo_young.GetValue()=="Peng_et_al_2006_e_Wirtz_et_al_2000":
            matrix_modulo_young_cortical, matrix_modulo_young_trabecular,=Density.MY_Peng_et_al_2006_e_Wirtz_et_al_2000(self.density_cortical,self.density_cancellous,self.spacing[0],self.spacing[1],self.spacing[2])
            numpy.save('modulo_young_cortical.npy', matrix_modulo_young_cortical)
            numpy.save('modulo_young_trabecular.npy', matrix_modulo_young_trabecular)
            mask =  numpy.zeros_like(matrix_modulo_young_cortical)
            self.cortical_viewer.SetInput(matrix_modulo_young_cortical, mask, self.spacing, 'ELASTICIDADE')
            self.trabecular_viewer.SetInput(matrix_modulo_young_trabecular, mask, self.spacing, 'ELASTICIDADE')




class Viewer(wx.Panel):

    polygon_points = vtk.vtkPoints()
    polygon = vtk.vtkPolygon()


    polygons = vtk.vtkCellArray()


    polygons = polygons

    ppolygon = vtk.vtkPolyData()
    ppolygon.SetPolys(polygons)
    ppolygon.SetPoints(polygon_points)
    idcell = ppolygon.InsertNextCell(polygon.GetCellType(), polygon.GetPointIds())

    ids = vtk.vtkIdList()

    extrude = vtk.vtkLinearExtrusionFilter()

    def __init__(self, prnt, orientation='AXIAL', tipo='NORMAL', titulo=None):
        wx.Panel.__init__(self, prnt)

        self.orientation = orientation
        self.slice_number = 0
        self.tipo = tipo

        self.image_input = None

        if titulo is None:
            self.titulo =  self.orientation
        else:
            self.titulo = titulo

        self.__init_gui()
        self.config_interactor()
        self.__bind_events_wx()
        self.__bind_events()

        # Apenas usado para densidade
        self.colorbar = None

        self.msize = 1
        self.mtype = "Max"

        self.ww = 255
        self.wl = 127

        self.nodes = None


        self.threshold = (300, 3033)


        self.npoints = 0

        self.islog = False
        self.ipolygon = []

        self.clicked = False

    def __init_gui(self):
        interactor = wxVTKRenderWindowInteractor(self, -1, size=self.GetSize())

        scroll = wx.ScrollBar(self, -1, style=wx.SB_VERTICAL)
        self.scroll = scroll


        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(wx.StaticText(self, -1, self.titulo))
        sizer.Add(interactor, 1, wx.EXPAND|wx.GROW)

        background_sizer = wx.BoxSizer(wx.HORIZONTAL)
        background_sizer.AddSizer(sizer, 1, wx.EXPAND|wx.GROW|wx.ALL, 2)
        background_sizer.Add(scroll, 0, wx.EXPAND|wx.GROW)
        self.SetSizer(background_sizer)
        background_sizer.Fit(self)

        self.Layout()
        self.Update()
        self.SetAutoLayout(1)

        self.interactor = interactor

    def __update_camera(self):
        orientation = self.orientation

        self.cam.SetFocalPoint(0, 0, 0)
        self.cam.SetViewUp(const.SLICE_POSITION[1][0][self.orientation])
        self.cam.SetPosition(const.SLICE_POSITION[1][1][self.orientation])
        self.cam.ParallelProjectionOn()


    def config_interactor(self):
        style = vtk.vtkInteractorStyleImage()
        self.style = style

        self.picker = vtk.vtkCellPicker()
        self.actor = None


        ren = vtk.vtkRenderer()
        ren.SetBackground((0, 0, 0))
        ren.SetLayer(0)

        self.ren = ren

        cam = ren.GetActiveCamera()
        cam.ParallelProjectionOn()

        ren2 = vtk.vtkRenderer()
        ren2.SetInteractive(0)

        ren2.SetActiveCamera(cam)
        ren2.SetLayer(1)

        interactor = self.interactor
        interactor.SetInteractorStyle(style)
        interactor.GetRenderWindow().SetNumberOfLayers(2)
        interactor.GetRenderWindow().AddRenderer(ren)
        interactor.GetRenderWindow().AddRenderer(ren2)
        interactor.SetPicker(self.picker)



        self.cam = ren.GetActiveCamera()
        self.ren = ren

        self.linhaActor1=vtk.vtkActor()
        self.linhaActor1.GetProperty().SetRepresentationToWireframe()
        self.linhaActor1.GetProperty().SetColor(1, 1, 1)
        self.linhaActor1.GetProperty().SetAmbient(1.0);
        self.linhaActor1.GetProperty().SetDiffuse(0.0);
        self.linhaActor1.GetProperty().SetSpecular(0.0);
        self.ren.AddActor(self.linhaActor1)

    def AddMarker(self, x, y, z):
        s = vtk.vtkSphereSource()
        s.SetCenter(x, y, z)
        s.SetRadius(self.spacing[0])

        m = vtk.vtkGlyph3DMapper()
        m.SetInputConnection(s.GetOutputPort())
        m.OrientOn()

        a = vtk.vtkActor()
        a.SetMapper(m)

        self.ren.AddActor(a)

    def SetPlistProject(self, pfile):
        """
        Parameters:
            pfile: A plist file preset.
        """
        preset = plistlib.readPlist(pfile)
        ncolours = len(preset['Blue'])
        prop = self.ww * 1.0 / ncolours

        init = self.wl - self.ww/2.0

        self.nodes = []
        for i in xrange(ncolours):
            r = preset['Red'][i]
            g = preset['Green'][i]
            b = preset['Blue'][i]
            v = init + i * prop
            node = clut_imagedata.Node(v, (r, g, b))

            self.nodes.append(node)

    def SetOrientation(self, orientation):
        self.orientation = orientation
        if self.orientation == 'AXIAL':
            max_slice_number = len(self.image_input)
        elif self.orientation == 'CORONAL':
            max_slice_number = self.image_input.shape[1]
        elif self.orientation == 'SAGITAL':
            max_slice_number = self.image_input.shape[2]

        self.scroll.SetScrollbar(wx.SB_VERTICAL, 1, max_slice_number,
                                 max_slice_number)
        self.__update_camera()
        self.ren.ResetCamera()



    def SetInput(self, m_input, mask, spacing, tipo='NORMAL'):
        self.image_input = m_input
        self.mask = mask
        self.spacing = spacing
        self.tipo=tipo
        if self.orientation == 'AXIAL':
            max_slice_number = len(m_input)
        elif self.orientation == 'CORONAL':
            max_slice_number = m_input.shape[1]
        elif self.orientation == 'SAGITAL':
            max_slice_number = m_input.shape[2]

        if self.tipo in ('DENSIDADE', 'ELASTICIDADE'):
            if self.colorbar is not None:
                self.ren.RemoveActor(self.colorbar)
            self.densidade_rmin = m_input.min()
            self.densidade_rmax = m_input.max()
            self.SetPlistProject('color_list/Jet.plist')
            self.colorbar = vtk.vtkScalarBarActor()
            self.ren.AddActor(self.colorbar)

        self.scroll.SetScrollbar(wx.SB_VERTICAL, 1, max_slice_number,
                                 max_slice_number)

        zs, ys, xs = m_input.shape

        xi = (1/3.0) * xs * spacing[0]
        xf = (2/3.0) * xs * spacing[0]

        yi = (1/3.0) * ys * spacing[1]
        yf = (2/3.0) * ys * spacing[1]

        zi = (1/3.0) * zs * spacing[2]
        zf = (2/3.0) * zs * spacing[2]

        cube = vtk.vtkCubeSource()
        cube.SetBounds(xi, xf, yi, yf, zi, zf)

        cam = self.ren.GetActiveCamera()

        self.plane1 = vtk.vtkPlane()
        self.plane1.SetOrigin(0.5,0.,0.)
        self.plane1.SetNormal(1, 0, 0)

        self.plane2 = vtk.vtkPlane()
        self.plane2.SetOrigin(0.51, 0., 0.)
        self.plane2.SetNormal(-1, 0, 0)

        planes = vtk.vtkPlaneCollection()
        planes.AddItem(self.plane1)
        planes.AddItem(self.plane2)

        self.clipperOutline = vtk.vtkClipClosedSurface()
        self.clipperOutline.SetInputData(self.extrude.GetOutput())
        self.clipperOutline.SetClippingPlanes(planes)
        self.clipperOutline.GenerateFacesOn()
        self.clipperOutline.GenerateOutlineOn()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.clipperOutline.GetOutput())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.8, 0, 0)
        actor.GetProperty().SetOpacity(0.6)

        self.ren.AddActor(actor)
        print "Added CUBE"

        self.SetSliceNumber(0)
        self.__update_camera()
        self.ren.ResetCamera()

    def __bind_events_wx(self):
        self.scroll.Bind(wx.EVT_SCROLL, self.OnScrollBar)

    def __bind_events(self):
        Publisher.subscribe(self.SetMipSize,
                                 'Set mip size')
        Publisher.subscribe(self.SetMipType,
                                 'Set mip type')
        Publisher.subscribe(self.SetThreshold,
                                 'Set threshold')
        Publisher.subscribe(self.UpdateRender,
                                 'Update render')
        Publisher.subscribe(self.UpdateWLWW,
                                 'Set wl ww')
        Publisher.subscribe(self.SetNodes,
                            'Set nodes')

    def SetMipSize(self, pubsub_evt):
        print "Set Mip, manolo"
        self.msize = pubsub_evt.data
        n = self.scroll.GetThumbPosition()
        try:
            self.SetSliceNumber(n)
        except:
            pass

    def SetMipType(self, pubsub_evt):
        self.mtype = pubsub_evt.data
        n = self.scroll.GetThumbPosition()
        try:
            self.SetSliceNumber(n)
        except:
            pass

    def SetThreshold(self, pubsub_evt):
        self.threshold = pubsub_evt.data


    def UpdateWLWW(self, pubsub_evt):
        self.wl, self.ww = pubsub_evt.data
        n = self.scroll.GetThumbPosition()
        try:
            self.SetSliceNumber(n)
        except:
            pass

    def SetNodes(self, pubsub_evt):
        self.nodes = pubsub_evt.data
        n = self.scroll.GetThumbPosition()
        try:
            self.SetSliceNumber(n)
        except:
            pass

    def UpdateRender(self, pubsub_evt):
        print self.orientation, "updating render"
        n = self.scroll.GetThumbPosition()
        self.SetSliceNumber(n)

    def OnScrollBar(self, evt):
        n = self.scroll.GetThumbPosition()
        print "Slice ->", n
        self.SetSliceNumber(n)

    def add_marker(self, obj, evt):
        mouse_x, mouse_y = self.interactor.GetEventPosition()
        self.picker.Pick(mouse_x, mouse_y, 0, self.ren)
        p_position = self.picker.GetPickPosition()
        position = self.actor.GetInput().FindPoint(p_position)
        n = self.scroll.GetThumbPosition()
        value = 255 if evt == 'LeftButtonPressEvent' else 0
        Publisher.sendMessage('Add marker', ((n,
                                                   position/self.image.GetDimensions()[0],
                                                   position%self.image.GetDimensions()[1]),
                                                 value))

    def SetSliceNumber(self, n):
        if self.image_input is None:
            return
        print self.orientation, n, self.msize, self.mtype
        if self.orientation == 'AXIAL':
            if self.mtype == "Max":
                n_array = numpy.array(self.image_input[n:n+self.msize].max(0))
            elif self.mtype == "Min":
                n_array = numpy.array(self.image_input[n:n+self.msize].min(0))
            elif self.mtype == "Mean":
                n_array = numpy.array(self.image_input[n:n+self.msize].mean(0))
            elif self.mtype == "Median":
                n_array = numpy.array(numpy.var(self.image_input[n:n+self.msize], 0))
            elif self.mtype == 'LMIP':
                #tmp_array = self.image_input[n:n+self.msize]
                #thresh_array = (tmp_array >= self.threshold[0]) & (tmp_array <= self.threshold[1])
                #argmax_array = thresh_array.argmax(0)
                #max_thresh_array = thresh_array.max(0)
                #max_array = tmp_array.max(0)


                #y, x = numpy.indices((argmax_array.shape))
                #n_array = tmp_array[argmax_array, y, x]
                #n_array[numpy.bitwise_not(max_thresh_array)] = max_array[numpy.bitwise_not(max_thresh_array)]
                tmp_array = self.image_input[n:n+self.msize]
                n_array = numpy.empty(shape=(tmp_array.shape[1],tmp_array.shape[2]), dtype=tmp_array.dtype)
                lmip.lmip(tmp_array, 0, self.threshold[0],
                     self.threshold[1], n_array)
            elif self.mtype == 'MIDA':
                tmp_array = self.image_input[n:n+self.msize]
                n_array = numpy.empty(shape=(tmp_array.shape[1],tmp_array.shape[2]), dtype=tmp_array.dtype)
                lmip.mida(tmp_array, 0, self.wl, self.ww, n_array)

                print "=================================="
                print ">>>", n_array.min(), n_array.max()
                print "=================================="

            else:
                n_array = numpy.array(self.image_input[n])

            mask = self.mask[n+1]

        elif self.orientation == 'CORONAL':
            if self.mtype == "Max":
                n_array = numpy.array(self.image_input[:, n:n+self.msize, :].max(1))
            elif self.mtype == "Min":
                n_array = numpy.array(self.image_input[:, n:n+self.msize, :].min(1))
            elif self.mtype == "Mean":
                n_array = numpy.array(self.image_input[:, n:n+self.msize, :].mean(1))
            elif self.mtype == "Median":
                n_array = numpy.array(numpy.var(self.image_input[:, n:n+self.msize, :], 1))
            elif self.mtype == 'LMIP':
                tmp_array = self.image_input[:, n:n+self.msize, :]
                #thresh_array = (tmp_array >= self.threshold[0]) & (tmp_array <= self.threshold[1])
                #argmax_array = thresh_array.argmax(1)
                #max_thresh_array = thresh_array.max(1)
                #max_array = tmp_array.max(1)


                #z, x = numpy.indices((argmax_array.shape))
                #n_array = tmp_array[z, argmax_array, x]
                #n_array[numpy.bitwise_not(max_thresh_array)] = max_array[numpy.bitwise_not(max_thresh_array)]
                n_array = numpy.empty(shape=(tmp_array.shape[0],tmp_array.shape[2]), dtype=tmp_array.dtype)
                lmip.lmip(tmp_array, 1, self.threshold[0],
                     self.threshold[1], n_array)
            elif self.mtype == 'MIDA':
                tmp_array = self.image_input[:, n:n+self.msize, :]
                n_array = numpy.empty(shape=(tmp_array.shape[0],tmp_array.shape[2]), dtype=tmp_array.dtype)
                lmip.mida(tmp_array, 1, self.wl, self.ww, n_array)
            else:
                n_array = numpy.array(self.image_input[:, n, :])

            mask = self.mask[1:, n+1, 1:]

        elif self.orientation == 'SAGITAL':
            if self.mtype == "Max":
                n_array = numpy.array(self.image_input[:, : , n:n+self.msize].max(2))
            elif self.mtype == "Min":
                n_array = numpy.array(self.image_input[:, : , n:n+self.msize].min(2))
            elif self.mtype == "Mean":
                n_array = numpy.array(self.image_input[:, : , n:n+self.msize].mean(2))
            elif self.mtype == "Median":
                n_array = numpy.array(numpy.var(self.image_input[:, : , n:n+self.msize], 2))
            elif self.mtype == 'LMIP':
                tmp_array = self.image_input[:, : , n:n+self.msize]
                #thresh_array = (tmp_array >= self.threshold[0]) & (tmp_array <= self.threshold[1])
                #argmax_array = thresh_array.argmax(2)
                #max_thresh_array = thresh_array.max(2)
                #max_array = tmp_array.max(2)


                #z, y = numpy.indices((argmax_array.shape))
                #n_array = tmp_array[z, y, argmax_array]
                #n_array[numpy.bitwise_not(max_thresh_array)] = max_array[numpy.bitwise_not(max_thresh_array)]
                n_array = numpy.empty(shape=(tmp_array.shape[0],tmp_array.shape[1]), dtype=tmp_array.dtype)
                lmip.lmip(tmp_array, 2, self.threshold[0],
                     self.threshold[1], n_array)
            elif self.mtype == 'MIDA':
                tmp_array = self.image_input[:, : , n:n+self.msize]
                n_array = numpy.empty(shape=(tmp_array.shape[0],tmp_array.shape[1]), dtype=tmp_array.dtype)
                lmip.mida(tmp_array, 2, self.wl, self.ww, n_array)
            else:
                n_array = numpy.array(self.image_input[:, : ,n])

            print n_array.shape, self.msize
            mask = self.mask[1:, 1:, n+1]

        n_shape = n_array.shape
        #mask = ((n_array >= self.threshold[0]) & (n_array <= self.threshold[1])).astype('uint8') * 255
        image = self.to_vtk(n_array, self.spacing, n, self.orientation)
        vmask = self.to_vtk(mask, self.spacing, n, self.orientation)

        print "O TIPO EH", self.tipo

        if self.tipo == 'DENSIDADE':
            self.image, lookuptable = self.do_ww_wl(image)
            self.image = self.do_blend(self.image, vmask)

            self.colorbar.SetTitle('g/cm3')
            self.colorbar.SetLookupTable(lookuptable);
            self.colorbar.SetOrientationToHorizontal()
            self.colorbar.SetWidth(.8);
            self.colorbar.SetHeight(.15);
            self.colorbar.SetPosition(0.05, 0.1);
            self.colorbar.SetLabelFormat("%.3g");
            self.colorbar.PickableOff();
            self.colorbar.VisibilityOn();

            if self.actor is None:
                self.actor = vtk.vtkImageActor()
                self.actor.PickableOn()
                self.ren.AddActor(self.actor)

            self.actor.SetInputData(self.image)
            self.actor.SetDisplayExtent(self.image.GetExtent())

        elif self.tipo == 'ELASTICIDADE':
            self.image, lookuptable = self.do_ww_wl(image)
            self.image = self.do_blend(self.image, vmask)

            self.colorbar.SetTitle('MPa')
            self.colorbar.SetLookupTable(lookuptable);
            self.colorbar.SetOrientationToHorizontal()
            self.colorbar.SetWidth(0.8);
            self.colorbar.SetHeight(0.15);
            self.colorbar.SetPosition(0.05, 0.1);
            self.colorbar.SetLabelFormat("%.3g");
            self.colorbar.PickableOff();
            self.colorbar.VisibilityOn();


            if self.actor is None:
                self.actor = vtk.vtkImageActor()
                self.actor.PickableOn()
                self.ren.AddActor(self.actor)


            self.actor.SetInputData(self.image)
            self.actor.SetDisplayExtent(self.image.GetExtent())

        else:
            self.image = self.do_ww_wl(image)
            self.image = self.do_blend(self.image, self.do_colour_mask(vmask))

            if self.actor is None:
                self.actor = vtk.vtkImageActor()
                self.actor.PickableOn()
                self.ren.AddActor(self.actor)

            self.actor.SetInputData(self.image)
            self.actor.SetDisplayExtent(self.image.GetExtent())

        self.__update_display_extent(self.image)
        self.interactor.Render()



    def __update_display_extent(self, image):
        self.actor.SetDisplayExtent(image.GetExtent())
        self.ren.ResetCameraClippingRange()

    def to_vtk(self, n_array, spacing, slice_number, orientation):
        try:
            dz, dy, dx = n_array.shape
        except ValueError:
            dy, dx = n_array.shape
            dz = 1

        v_image = numpy_support.numpy_to_vtk(n_array.flat)

        if orientation == 'AXIAL':
            extent = (0, dx -1, 0, dy -1, slice_number, slice_number + dz - 1)
        elif orientation == 'SAGITAL':
            dx, dy, dz = dz, dx, dy
            extent = (slice_number, slice_number + dx - 1, 0, dy - 1, 0, dz - 1)
        elif orientation == 'CORONAL':
            dx, dy, dz = dx, dz, dy
            extent = (0, dx - 1, slice_number, slice_number + dy - 1, 0, dz - 1)

        # Generating the vtkImageData
        image = vtk.vtkImageData()
        image.SetOrigin(0, 0, 0)
        image.SetSpacing(spacing)
        #image.SetNumberOfScalarComponents(1)
        image.SetExtent(0, dx -1, 0, dy -1, 0, dz - 1)
        image.SetDimensions(dx, dy, dz)
        image.SetExtent(extent)
        #image.SetScalarType(numpy_support.get_vtk_array_type(n_array.dtype))
        #image.AllocateScalars()
        image.AllocateScalars(numpy_support.get_vtk_array_type(n_array.dtype),1)
        image.GetPointData().SetScalars(v_image)
        #image.Update()

        image_copy = vtk.vtkImageData()
        image_copy.DeepCopy(image)
        #image_copy.Update()

        return image_copy


    def do_ww_wl(self, image):
        if self.nodes and self.tipo not in ('DENSIDADE', 'ELASTICIDADE'):

            snodes = sorted(self.nodes)

            lut = vtk.vtkWindowLevelLookupTable()
            lut.SetWindow(self.ww)
            lut.SetLevel(self.wl)

            lut.Build()

            for i, n in enumerate(self.nodes):
                r, g, b = n.colour
                lut.SetTableValue(i, r/255.0, g/255.0, b/255.0, 1.0)

            colorer = vtk.vtkImageMapToColors()
            colorer.SetInputData(image)

            colorer.SetLookupTable(lut)
            colorer.SetOutputFormatToRGB()
            colorer.Update()

        elif self.tipo in ('DENSIDADE', 'ELASTICIDADE'):
            lut = vtk.vtkWindowLevelLookupTable()
            lut.SetWindow(self.ww)
            lut.SetLevel(self.wl)
            lut.SetTableRange(self.densidade_rmin, self.densidade_rmax)

            lut.Build()

            crange = self.densidade_rmax - self.densidade_rmin
            prop = crange / len(self.nodes)

            for i, n in enumerate(self.nodes):
                r, g, b = n.colour
                lut.SetTableValue(i, r/255.0, g/255.0, b/255.0, 1.0)

            colorer = vtk.vtkImageMapToColors()
            colorer.SetInputData(image)

            colorer.SetLookupTable(lut)
            colorer.SetOutputFormatToRGB()
            colorer.Update()
            return colorer.GetOutput(), lut

        else:
            colorer = vtk.vtkImageMapToWindowLevelColors()
            colorer.SetInputData(image)
            colorer.SetWindow(self.ww)
            colorer.SetLevel(self.wl)
            colorer.SetOutputFormatToRGB()
            colorer.Update()

        return colorer.GetOutput()

    def do_blend(self, imagedata, mask):
        # blend both imagedatas, so it can be inserted into viewer
        print "Blending Spacing", imagedata.GetSpacing(), mask.GetSpacing()

        blend_imagedata = vtk.vtkImageBlend()
        blend_imagedata.SetBlendModeToNormal()
        # blend_imagedata.SetOpacity(0, 1.0)
        blend_imagedata.SetOpacity(1, 0.8)
        blend_imagedata.SetInputData(imagedata)
        blend_imagedata.AddInputData(mask)
        blend_imagedata.Update()


        # return colorer.GetOutput()

        return blend_imagedata.GetOutput()

    def __create_background(self, imagedata):

        thresh_min, thresh_max = imagedata.GetScalarRange()

        # map scalar values into colors
        lut_bg = vtk.vtkLookupTable()
        lut_bg.SetTableRange(thresh_min, thresh_max)
        lut_bg.SetSaturationRange(0, 0)
        lut_bg.SetHueRange(0, 0)
        lut_bg.SetValueRange(0, 1)
        lut_bg.Build()

        # map the input image through a lookup table
        img_colours_bg = vtk.vtkImageMapToColors()
        img_colours_bg.SetOutputFormatToRGBA()
        img_colours_bg.SetLookupTable(lut_bg)
        img_colours_bg.SetInputData(imagedata)
        img_colours_bg.Update()

        return img_colours_bg.GetOutput()

    def do_colour_mask(self, imagedata):
        scalar_range = int(imagedata.GetScalarRange()[1])
        r,g,b = 0, 1, 0

        # map scalar values into colors
        lut_mask = vtk.vtkLookupTable()
        lut_mask.SetNumberOfColors(255)
        lut_mask.SetHueRange(const.THRESHOLD_HUE_RANGE)
        lut_mask.SetSaturationRange(1, 1)
        lut_mask.SetValueRange(0, 1)
        lut_mask.SetNumberOfTableValues(256)
        lut_mask.SetTableValue(0, 0, 0, 0, 0.0)
        lut_mask.SetTableValue(1, 0, 0, 0, 0.0)
        lut_mask.SetTableValue(2, 0, 0, 0, 0.0)
        lut_mask.SetTableValue(255, r, g, b, 1.0)
        lut_mask.SetRampToLinear()
        lut_mask.Build()
        # self.lut_mask = lut_mask

        # map the input image through a lookup table
        img_colours_mask = vtk.vtkImageMapToColors()
        img_colours_mask.SetLookupTable(lut_mask)
        img_colours_mask.SetOutputFormatToRGBA()
        img_colours_mask.SetInputData(imagedata)
        img_colours_mask.Update()
        # self.img_colours_mask = img_colours_mask

        return img_colours_mask.GetOutput()

    def __create_mask_threshold(self, imagedata):
        thresh_min, thresh_max = self.threshold

        # flexible threshold
        img_thresh_mask = vtk.vtkImageThreshold()
        img_thresh_mask.SetInValue(const.THRESHOLD_INVALUE)
        img_thresh_mask.SetInputData(imagedata)
        img_thresh_mask.SetOutValue(const.THRESHOLD_OUTVALUE)
        img_thresh_mask.ThresholdBetween(float(thresh_min), float(thresh_max))
        img_thresh_mask.ReplaceInOn()
        img_thresh_mask.ReplaceOutOn()
        img_thresh_mask.SetInValue(255)
        img_thresh_mask.SetOutValue(0)
        img_thresh_mask.SetOutputScalarTypeToUnsignedChar()
        img_thresh_mask.Update()

        return img_thresh_mask.GetOutput()


    def desenha_linhas(self, xi,xf,yi,yf,zi,zf):
        xbi, xbf, ybi, ybf, zbi, zbf = self.actor.GetBounds()

        xli=self.spacing[0]*xi
        xlf=self.spacing[0]*xf

        yli=self.spacing[1]*yi
        ylf=self.spacing[1]*yf

        zli=self.spacing[2]*zi
        zlf=self.spacing[2]*zf


        linha=vtk.vtkCubeSource()
        linha.SetBounds(xli,xlf,yli,ylf, zli,zlf)

        maperlinha=vtk.vtkPolyDataMapper()
        maperlinha.SetInputData(linha.GetOutput())
        self.linhaActor1.SetMapper(maperlinha)


        self.interactor.Render()







class SurfaceViewer(wx.Panel):
    """docstring for SurfaceViewer"""
    def __init__(self, parent):
        super(SurfaceViewer, self).__init__(-1, parent)

class Surface(multiprocessing.Process):
    def __init__(self, image, matrix, roi, spacing, threshold, output_dir, queue_output):
        multiprocessing.Process.__init__(self)

        self.matrix_info = matrix
        self.image_info = image

        self.roi = roi
        self.spacing = spacing
        self.threshold = threshold
        self.output_dir = output_dir
        self.queue_output = queue_output

    def run(self):
        self.matrix = numpy.memmap(self.matrix_info['filename'], mode='r+',
                                   shape=self.matrix_info['shape'],
                                   dtype=self.matrix_info['type'])

        self.image = numpy.memmap(self.image_info['filename'], mode='r+',
                                   shape=self.image_info['shape'],
                                   dtype=self.image_info['type'])

        n_array = numpy.array(self.image[self.roi])
        n_mask = numpy.array(self.matrix[self.roi])
        n_array[n_mask == 2] = -3034
        n_mask = ndimage.gaussian_filter(n_mask, (1, 1, 1))
        image = self.to_vtk(n_mask)

        output = tempfile.mktemp(prefix=str(self.ident), suffix='.vti', dir=self.output_dir)

        w = vtk.vtkXMLImageDataWriter()
        w.SetInputData(image)
        w.SetFileName(output)
        w.Write()

        print "Written Imagedata", output

        output = self.generate_surface(image)
        self.queue_output.put(output)

    def to_vtk(self, n_array):
        dz, dy, dx = n_array.shape
        n_array.shape = dx * dy * dz

        v_image = numpy_support.numpy_to_vtk(n_array)

        # Generating the vtkImageData
        image = vtk.vtkImageData()
        image.SetDimensions(dx, dy, dz)
        image.SetOrigin(0, 0, self.roi.start * self.spacing[2])
        image.SetSpacing(self.spacing)
        image.SetNumberOfScalarComponents(1)
        image.SetExtent(0, dx -1, 0, dy -1, 0, dz - 1)
        image.SetScalarType(numpy_support.get_vtk_array_type(n_array.dtype))
        image.AllocateScalars()
        image.GetPointData().SetScalars(v_image)
        image.Update()

        image_copy = vtk.vtkImageData()
        image_copy.DeepCopy(image)
        image_copy.Update()

        return image_copy

    def generate_surface(self, image):
        t_min, t_max = self.threshold
        mcubes = vtk.vtkContourFilter()
        mcubes.GenerateValues(3, t_min, t_max)
        mcubes.SetInputData(image)
        mcubes.UseScalarTreeOn()
        mcubes.ComputeGradientsOn()
        mcubes.ComputeNormalsOn()
        mcubes.ComputeScalarsOn()
        mcubes.Update()


        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputData(mcubes.GetOutput())
        smoother.SetNumberOfIterations(2)
        smoother.SetFeatureAngle(30)
        smoother.FeatureEdgeSmoothingOn()
        smoother.BoundarySmoothingOn()
        smoother.Update()
        smoother.GetOutput().ReleaseDataFlagOn()

        n_input = smoother

        print "Normals"
        normals = vtk.vtkPolyDataNormals()
        # normals.GetOutput().ReleaseDataFlagOff()
        normals.SetInputData(n_input.GetOutput())
        normals.SetFeatureAngle(80)
        normals.Update()

        print "Stripper"
        stripper = vtk.vtkStripper()
        stripper.SetInputData(normals.GetOutput())
        # stripper.GetOutput().ReleaseDataFlagOff()
        stripper.Update()

        output = tempfile.mktemp(prefix=str(self.ident), suffix='.vtp', dir=self.output_dir)

        stlBinary = vtk.vtkXMLPolyDataWriter()
        stlBinary.SetInputData(smoother.GetOutput())
        stlBinary.SetFileName('/tmp/superficie_do_camboja.vtp')
        stlBinary.Write()

        print output

        return output


class App(wx.App):
    def OnInit(self):
        self.frame = Window(None)
        self.frame.Center()
        self.SetTopWindow(self.frame)
        return True

if __name__ == '__main__':
    app = App(0)
    app.MainLoop()

