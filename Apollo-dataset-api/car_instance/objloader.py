from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.GLU import *


import numpy as np
import pygame
import OpenGL

'''
the original is here https://www.pygame.org/wiki/OBJFileLoader
@2018-1-2 author chj
change for easy use
'''


def MTL(fdir, filename):
    contents = {}
    mtl = None
    for line in open(fdir + filename, "r"):
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue
        if values[0] == 'newmtl':
            mtl = contents[values[1]] = {}
        elif mtl is None:
            raise ValueError("mtl file doesn't start with newmtl stmt")
        elif values[0] == 'map_Kd':
            # load the texture referred to by this declaration
            mtl[values[0]] = values[1]
            surf = pygame.image.load(fdir + mtl['map_Kd'])
            image = pygame.image.tostring(surf, 'RGBA', 1)
            ix, iy = surf.get_rect().size
            texid = mtl['texture_Kd'] = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texid)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                            GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                            GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0, GL_RGBA,
                         GL_UNSIGNED_BYTE, image)
        else:
            #mtl[values[0]] = map(float, values[1:])

            mtl[values[0]] = [float(x) for x in values[1:4]]
    return contents


class OBJ:
    def __init__(self, fdir, filename, swapyz=False):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []

        self.mtl = None

        material = None
        for line in open(fdir + filename, "r"):
            if line.startswith('#'):
                continue
            values = line.split()
            if not values:
                continue
            if values[0] == 'v':
                #v = map(float, values[1:4])
                v = [float(x) for x in values[1:4]]
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == 'vn':
                #v = map(float, values[1:4])
                v = [float(x) for x in values[1:4]]
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                v = [float(x) for x in values[1:3]]

                self.texcoords.append(v)
            elif values[0] in ('usemtl', 'usemat'):
                material = values[1]
            elif values[0] == 'mtllib':
                # print(values[1])
                #self.mtl = MTL(fdir,values[1])
                self.mtl = [fdir, values[1]]
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                self.faces.append((face, norms, texcoords, material))

    def create_bbox(self):
        # self.vertices is not None
        ps = np.array(self.vertices)
        vmin = ps.min(axis=0)
        vmax = ps.max(axis=0)

        self.bbox_center = (vmax + vmin) / 2
        self.bbox_half_r = np.max(vmax - vmin) / 2

    def create_gl_list(self):
        if self.mtl is not None:
            print(self.mtl, "---")
            self.mtl = MTL(*self.mtl)

        self.gl_list = glGenLists(1)
        glNewList(self.gl_list, GL_COMPILE)
        glEnable(GL_TEXTURE_2D)
        glFrontFace(GL_CCW)
        # glCullFace(GL_BACK)
        # glEnable(GL_CULL_FACE)

        for face in self.faces:
            vertices, normals, texture_coords, material = face

            mtl = self.mtl[material]
            if 'texture_Kd' in mtl:
                # use diffuse texmap
                glBindTexture(GL_TEXTURE_2D, mtl['texture_Kd'])
            else:
                # just use diffuse colour
                # print(mtl['Kd'],"----")
                glColor(*mtl['Kd'])

            glBegin(GL_POLYGON)
            for i in range(len(vertices)):
                if normals[i] > 0:
                    glNormal3fv(self.normals[normals[i] - 1])
                if texture_coords[i] > 0:
                    glTexCoord2fv(self.texcoords[texture_coords[i] - 1])
                glVertex3fv(self.vertices[vertices[i] - 1])
            glEnd()
        glDisable(GL_TEXTURE_2D)
        glEndList()


'''
@2018-3-13
主要处理
v x y z r g b
f a b c
'''


class CHJ_tiny_obj:
    def __init__(self, fdir, filename, swapyz=False):
        if filename is None:
            return
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        self.v_colors = {}

        self.mtl = None

        fname = fdir + filename
        for line in open(fname, "r"):
            if line.startswith('#'):
                continue
            values = line.split()
            if not values:
                continue
            if values[0] == 'v':
                # v = map(float, values[1:4])
                v = [float(x) for x in values[1:4]]
                if swapyz:
                    v = [v[0], v[2], v[1]]
                self.vertices.append(v)
                if len(values) == 7:
                    c = [float(x) for x in values[4:7]]
                    # self.v_colors.append(c)
                    self.v_colors[len(self.vertices) - 1] = c
            elif values[0] == 'vn':
                # v = list(map(float, values[1:4]))
                v = [float(x) for x in values[1:4]]
                if swapyz:
                    v = [v[0], v[2], v[1]]
                self.normals.append(v)
            elif values[0] == 'vt':
                v = [float(x) for x in values[1:3]]
            elif values[0] == 'f':
                v = [int(x) for x in values[1:4]]
                self.faces.append(v)

    def set_V_T_F(self, v, t, f):
        self.vertices = v
        self.v_colors = t
        self.faces = f

    def create_bbox(self):
        # self.vertices is not None
        ps = np.array(self.vertices)
        vmin = ps.min(axis=0)
        vmax = ps.max(axis=0)

        self.bbox_center = (vmax + vmin) / 2
        self.bbox_half_r = np.max(vmax - vmin) / 2

    def create_gl_list(self):
        self.gl_list = glGenLists(1)
        glNewList(self.gl_list, GL_COMPILE)
        glFrontFace(GL_CCW)
        # glCullFace(GL_BACK)
        # glEnable(GL_CULL_FACE)
        # print(self.faces)
        # print(self.v_colors)

        for face in self.faces:
            # print(face)
            glBegin(GL_TRIANGLES)
            # glBegin(GL_POLYGON)
            for vid in face:
                vid -= 1
                # if self.v_colors[vid] is not None:
                #     glColor3f(*self.v_colors[vid])
                glVertex3fv(self.vertices[vid])  # 看好后面加了个v，只用传地址就行了
            glEnd()

        glEndList()