# coding=utf-8

import numpy
import pylab

# calculo densidade metodo taylor

def Taylor_W_R_Roland(matrix_imagem, sx,sy,sz):
    matrix_density_cortical_bone=numpy.zeros(matrix_imagem.shape, dtype='float32')
    matrix_density_cancellous_bone=numpy.zeros(matrix_imagem.shape, dtype='float32')

    #volume
    v = (sx * sy * sz) / 1000.0

    matrix_density_cortical_bone[(matrix_imagem >= 400) & (matrix_imagem <= 2500)] = (4.64*10**(-4) )* matrix_imagem[(matrix_imagem >= 400) & (matrix_imagem<=2500)] + 1.0
    matrix_density_cancellous_bone[(matrix_imagem > 100) & (matrix_imagem < 400)] = (4.64*10**(-4) ) * matrix_imagem[(matrix_imagem > 100) & (matrix_imagem < 400)] + 1.0

    return matrix_density_cortical_bone, matrix_density_cancellous_bone


# calculo de densidade metodo pedro 200

def densidade_Pedro_2000(matrix_imagem, sx,sy,sz):
    matrix_density_cortical_bone=numpy.zeros(matrix_imagem.shape, dtype='float32')
    matrix_density_cancellous_bone=numpy.zeros(matrix_imagem.shape, dtype='float32')

    v = (sx * sy * sz) / 1000.0
    matrix_density_cortical_bone[(matrix_imagem >= 400) & (matrix_imagem <=2500)] = 0.001 * matrix_imagem[(matrix_imagem >= 400) & (matrix_imagem <=2500)] + 1.19
    matrix_density_cancellous_bone[(matrix_imagem > 100) & (matrix_imagem < 400)] = 0.001 * matrix_imagem[(matrix_imagem > 100) & (matrix_imagem < 400)] + 1.19

    return matrix_density_cortical_bone*v*1000, matrix_density_cancellous_bone*v*1000



# calculo dos modulos de Elasticidade para cada metodo

# modelo para modulo de young Carter e Haynes, 1977

def MY_Carter_Haynes_1977(matrix_density_cortical_bone, sx,sy,sz):

    matrix_modulo_young_cortical=numpy.zeros(matrix_density_cortical_bone.shape,dtype='float32')

    #cortical
    matrix_modulo_young_cortical=3790*matrix_density_cortical_bone**3

    return matrix_modulo_young_cortical


#modelo para modulo de young Rice et al, 1988

def MY_Rice_et_al_1988(matrix_density_cancellous_bone, sx,sy,sz):

    matrix_modulo_young_trabecular=numpy.zeros(matrix_density_cancellous_bone.shape,dtype='float32')

    #trabecular
    matrix_modulo_young_trabecular=900*matrix_density_cancellous_bone**2 + 60

    return matrix_modulo_young_trabecular

#modelo para modulo de young Rho et al, 1995

def MY_Rho_et_al_1995(matrix_density_cortical_bone, matrix_density_cancellous_bone, sx,sy,sz):

    matrix_modulo_young_cortical=numpy.zeros(matrix_density_cortical_bone.shape,dtype='float32')
    matrix_modulo_young_trabecular=numpy.zeros(matrix_density_cancellous_bone.shape,dtype='float32')


    #cortical
    matrix_modulo_young_cortical=4560*matrix_density_cortical_bone**(3.31)

    # Trabecular
    matrix_modulo_young_trabecular=4607.1*matrix_density_cancellous_bone**1.30

    return  matrix_modulo_young_cortical, matrix_modulo_young_trabecular


#modelo para modulo de young  Morgan et al, 2003

def MY_Morgan_et_al_2003(matrix_density_cortical_bone,matrix_density_cancellous_bone, sx,sy,sz):

    matrix_modulo_young_cortical=numpy.zeros(matrix_density_cortical_bone.shape,dtype='float32')
    matrix_modulo_young_trabecular=numpy.zeros(matrix_density_cancellous_bone.shape,dtype='float32')

    # cortical
    matrix_modulo_young_cortical=15010*matrix_density_cortical_bone**2.18

    #trabecular
    matrix_modulo_young_trabecular=6850*matrix_density_cancellous_bone**1.49

    return  matrix_modulo_young_cortical, matrix_modulo_young_trabecular



#modelo para modulo de young Keller, 1994

def MY_Keller_1994(matrix_density_cortical_bone, sx,sy,sz):

    matrix_modulo_young_cortical=numpy.zeros(matrix_density_cortical_bone.shape,dtype='float32')

    #cotical

    matrix_modulo_young_cortical=10500*matrix_density_cortical_bone**2.3


#modelo para modulo de young Peng et al, 2006 e Wirtz et al, 2000

def MY_Peng_et_al_2006_e_Wirtz_et_al_2000(matrix_density_cortical_bone,matrix_density_cancellous_bone, sx,sy,sz):

    matrix_modulo_young_cortical=numpy.zeros(matrix_density_cortical_bone.shape,dtype='float32')
    matrix_modulo_young_trabecular=numpy.zeros(matrix_density_cancellous_bone.shape,dtype='float32')

    # cortical
    matrix_modulo_young_cortical=2065*matrix_density_cortical_bone**3.09

    #trabecular
    matrix_modulo_young_trabecular=1904*matrix_density_cancellous_bone**1.647

    return  matrix_modulo_young_cortical, matrix_modulo_young_trabecular



























