from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from PIL import Image, ImageOps, ImageDraw
from sklearn.decomposition import PCA
from numba import jit
import sys
import math
import numpy as np
import time


# parametrii inițiali

# oglinzi 1, 2
concava = True
R = 376.0
D = 200.0
d_gura = 60.0

concava2 = True
R2 = 129.0
D2 = 50.0

dDD = 139.0

Bf = -20.0

# senzor (in mm)
plane = 35.0
pixel_size = 0.1

# diferente finite
l_triunghi = 3.0
l_triunghi2 = 2

N_THETA = 40
M_PHI = 80

# control vizualizare
zoom = 30.0
rotate_x, rotate_y = 0.0, 0.0
last_x, last_y = 0, 0
mouse_down = False

@jit(nopython=True)
def normalize(v):
    return v / np.linalg.norm(v)

@jit(nopython=True)
def compute_normal(A, B, C):
    AB = B - A
    AC = C - A
    N = np.cross(AB, AC)
    return normalize(N)

@jit(nopython=True)
def compute_triangle_center(v0, v1, v2):
    v0, v1, v2 = np.array(v0), np.array(v1), np.array(v2)
    # Centru geometric
    center = (v0 + v1 + v2) / 3.0
    
    return center

def generate_spherical_cap_triangles(R, D, N_theta, M_phi, concave=True, d=0.0):
    """Generează lista de triunghiuri pentru calota sferică cu eventuală gaură centrală de diametru d"""
    if D > 2 * R:
        raise ValueError("Diametrul calotei D nu poate fi mai mare decât 2R")
    if d < 0 or d > D:
        raise ValueError("Diametrul găurii d trebuie să fie între 0 și D")

    triangles = []
    theta_max = math.asin(D / (2 * R))
    sign = -1 if concave else 1
    r_min = d / 2.0  # raza găurii

    for i in range(N_theta):
        theta1 = theta_max * i / N_theta
        theta2 = theta_max * (i + 1) / N_theta

        for j in range(M_phi):
            phi1 = 2 * math.pi * j / M_phi
            phi2 = 2 * math.pi * (j + 1) / M_phi

            p1 = (R * math.sin(theta1) * math.cos(phi1),
                  R * math.sin(theta1) * math.sin(phi1),
                  sign * (R - R * math.cos(theta1)))

            p2 = (R * math.sin(theta2) * math.cos(phi1),
                  R * math.sin(theta2) * math.sin(phi1),
                  sign * (R - R * math.cos(theta2)))

            p3 = (R * math.sin(theta2) * math.cos(phi2),
                  R * math.sin(theta2) * math.sin(phi2),
                  sign * (R - R * math.cos(theta2)))

            p4 = (R * math.sin(theta1) * math.cos(phi2),
                  R * math.sin(theta1) * math.sin(phi2),
                  sign * (R - R * math.cos(theta1)))

            # funcție ajutătoare — verifică dacă punctul e în afara găurii
            @jit(nopython=True)
            def is_outside_hole(p):
                r = math.hypot(p[0], p[1])
                return r >= r_min

            # păstrăm triunghiurile doar dacă toate vârfurile sunt în afara găurii
            if all(is_outside_hole(p) for p in (p1, p2, p3)):
                triangles.append((p1, p2, p3))
            if all(is_outside_hole(p) for p in (p1, p3, p4)):
                triangles.append((p1, p3, p4))

    return triangles

def generate_spherical_cap_triangles_uniform(R, D, N_theta, M_phi, concave=True, d=0.0):
    """Generează triunghiuri pe calotă cu densitate triunghi uniformă (în suprafață)"""
    if D > 2 * R:
        raise ValueError("Diametrul calotei D nu poate fi mai mare decât 2R")
    if d < 0 or d > D:
        raise ValueError("Diametrul găurii d trebuie să fie între 0 și D")

    triangles = []
    theta_max = math.asin(D / (2 * R))
    sign = -1 if concave else 1
    r_min = d / 2.0  # raza găurii

    cos_theta_max = math.cos(theta_max)
    # Funcție pentru θ uniform pe aria suprafeței
    def theta_u(i):
        u = i / N_theta
        cos_theta = 1 - u * (1 - cos_theta_max)
        return math.acos(cos_theta)

    for i in range(N_theta):
        theta1 = theta_u(i)
        theta2 = theta_u(i + 1)

        for j in range(M_phi):
            phi1 = 2 * math.pi * j / M_phi
            phi2 = 2 * math.pi * (j + 1) / M_phi

            p1 = (R * math.sin(theta1) * math.cos(phi1),
                  R * math.sin(theta1) * math.sin(phi1),
                  sign * (R - R * math.cos(theta1)))

            p2 = (R * math.sin(theta2) * math.cos(phi1),
                  R * math.sin(theta2) * math.sin(phi1),
                  sign * (R - R * math.cos(theta2)))

            p3 = (R * math.sin(theta2) * math.cos(phi2),
                  R * math.sin(theta2) * math.sin(phi2),
                  sign * (R - R * math.cos(theta2)))

            p4 = (R * math.sin(theta1) * math.cos(phi2),
                  R * math.sin(theta1) * math.sin(phi2),
                  sign * (R - R * math.cos(theta1)))

            def is_outside_hole(p):
                r = math.hypot(p[0], p[1])
                return r >= r_min

            if all(is_outside_hole(p) for p in (p1, p2, p3)):
                triangles.append((p1, p2, p3))
            if all(is_outside_hole(p) for p in (p1, p3, p4)):
                triangles.append((p1, p3, p4))

    return triangles

def generate_triangular_grid(R, D, d=0.0, concave=True, edge_length=0.5):
    """
    Generează triunghiuri echilaterale pe toată calota sferică (nu doar jumătate),
    cu gaură centrală d, rază R și diametru calotă D.
    """
    sign = -1 if concave else 1
    r_max = D / 2
    r_min = d / 2

    dx = edge_length
    dy = edge_length * math.sqrt(3)/2

    # Generăm punctele într-un grid triunghiular hexagonal care acoperă
    # întreg cercul de raza r_max
    grid_points = []

    # Vom genera rânduri de la y = -r_max până la y = r_max
    iy_min = int(-r_max / dy) - 1
    iy_max = int(r_max / dy) + 1

    for iy in range(iy_min, iy_max + 1):
        y = iy * dy
        # offset pe x pentru rânduri pare/impare
        offset = 0 if (iy % 2 == 0) else dx / 2
        row = []
        ix_min = int(-r_max / dx) - 1
        ix_max = int(r_max / dx) + 1
        for ix in range(ix_min, ix_max + 1):
            x = ix * dx + offset
            r_xy = math.hypot(x, y)
            if r_xy <= r_max and r_xy >= r_min:
                # calculăm z pe calotă
                z = sign * (R - math.sqrt(max(R*R - x*x - y*y, 0)))
                row.append((x, y, z))
        if row:
            grid_points.append(row)

    # Construim triunghiurile similar ca înainte
    triangles = []
    for i in range(len(grid_points)-1):
        row1 = grid_points[i]
        row2 = grid_points[i+1]
        min_len = min(len(row1), len(row2))
        for j in range(min_len - 1):
            if i % 2 == 0:
                triangles.append((row1[j], row2[j], row2[j+1]))
                if j + 1 < len(row1):
                    triangles.append((row1[j], row2[j+1], row1[j+1]))
            else:
                triangles.append((row1[j], row2[j], row1[j+1]))
                if j + 1 < len(row2):
                    triangles.append((row1[j+1], row2[j], row2[j+1]))

    return triangles


def generate_uniform_triangular_mesh(R, D, d=0.0, concave=True, edge_length=0.5):
    """
    Generează triunghiuri echilaterale pe calota sferică de rază R,
    diametru D, gaură centrală d (d=0 înseamnă fără gaură),
    edge_length = latura triunghiurilor în planul XOY.
    concave: True pentru calotă concavă, False pentru convexă.
    Returnează lista de triunghiuri (fiecare triunghi = 3 puncte (x,y,z)).
    """
    sign = -1 if concave else 1
    r_max = D / 2
    r_min = d / 2

    dx = edge_length
    dy = edge_length * math.sqrt(3) / 2

    # Generăm punctele în grid triunghiular hexagonal
    grid_points = []  # listă de rânduri
    iy_min = int(-r_max / dy) - 1
    iy_max = int(r_max / dy) + 1

    for iy in range(iy_min, iy_max + 1):
        y = iy * dy
        offset = 0 if (iy % 2 == 0) else dx / 2
        row = []
        ix_min = int(-r_max / dx) - 1
        ix_max = int(r_max / dx) + 1
        for ix in range(ix_min, ix_max + 1):
            x = ix * dx + offset
            r_xy = math.hypot(x, y)
            # Excludem punctele în afara calotei sau în gaura centrală
            if r_xy <= r_max and r_xy >= r_min:
                # Proiectăm pe sferă
                z = sign * (R - math.sqrt(max(R*R - x*x - y*y, 0)))
                row.append((x, y, z))
            else:
                row.append(None)  # punct invalid / găură sau exterior
        grid_points.append(row)

    # Generăm triunghiurile: doar dacă toate cele 3 vârfuri sunt valide (nu None)
    triangles = []
    for i in range(len(grid_points) - 1):
        row1 = grid_points[i]
        row2 = grid_points[i + 1]
        len1 = len(row1)
        len2 = len(row2)
        for j in range(min(len1, len2) - 1):
            # rând par: triunghiuri cu offset
            if i % 2 == 0:
                #N1 = compute_normal(row1[j], row2[j], row2[j + 1])
                tri1 = (row1[j], row2[j], row2[j + 1])
               # N2 = compute_normal(row1[j], row2[j + 1], row1[j + 1])
                tri2 = (row1[j], row2[j + 1], row1[j + 1]) if (j + 1) < len1 else None
            else:
                #N1 = compute_normal(row1[j], row2[j], row1[j + 1])
                tri1 = (row1[j], row2[j], row1[j + 1]) if (j + 1) < len1 else None
                #N2 = compute_normal(row1[j + 1], row2[j], row2[j + 1])
                tri2 = (row1[j + 1], row2[j], row2[j + 1]) if (j + 1) < len2 else None

            for tri in [tri1, tri2]:
                if tri and all(v is not None for v in tri):
                    triangles.append(tri)

    return triangles

def generate_init_rays(triangles):
    rays = []
    intermediareRays = []
    
    for tri in triangles:
        p1 = (np.array(tri[0]) + np.array(tri[1]) + np.array(tri[2])) / 3.0
        #p1[2] = 0.0
        p2 = (np.array(tri[0]) + np.array(tri[1]) + np.array(tri[2])) / 3.0
        p2[2] = 999999.99
        rays.append((p1, p2))
        ray_out = reflect_ray_to_plane(p1, p2, tri, -1.5 * dDD)
        if ray_out is not None:
            intermediareRays.append(ray_out)
                
    # return rays[:200]         
    return intermediareRays

@jit(nopython=True)
def point_in_triangle(P, A, B, C):
    v0 = C - A
    v1 = B - A
    v2 = P - A

    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom

    return (u >= 0) and (v >= 0) and (u + v <= 1)

def reflect_ray_to_plane(ray_start, ray_end, triangle_vertices, z_target):
    A, B, C = [np.array(v) for v in triangle_vertices]
    P0 = np.array(ray_start)
    P1 = np.array(ray_end)

    D = P1 - P0
    D = D / np.linalg.norm(D)

    N = np.cross(B - A, C - A)
    N = N / np.linalg.norm(N)

    denom = np.dot(N, D)
    if np.abs(denom) < 1e-6:
        return None  # rază paralelă cu planul

    d = np.dot(N, A)
    t = (d - np.dot(N, P0)) / denom
    if t < 0:
        return None

    P = P0 + t * D

    # def point_in_triangle(P, A, B, C):
    #     v0 = C - A
    #     v1 = B - A
    #     v2 = P - A

    #     dot00 = np.dot(v0, v0)
    #     dot01 = np.dot(v0, v1)
    #     dot02 = np.dot(v0, v2)
    #     dot11 = np.dot(v1, v1)
    #     dot12 = np.dot(v1, v2)

    #     invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
    #     u = (dot11 * dot02 - dot01 * dot12) * invDenom
    #     v = (dot00 * dot12 - dot01 * dot02) * invDenom

    #     return (u >= 0) and (v >= 0) and (u + v <= 1)

    if not point_in_triangle(P, A, B, C):
        return None

    R = D - 2 * np.dot(D, N) * N
    R = R / np.linalg.norm(R)

    if np.abs(R[2]) < 1e-6:
        return None  # rază reflectată paralelă cu planul z=z_target

    t_reflect = (z_target - P[2]) / R[2]
    if t_reflect < 0:
        return None  # intersecția e în spatele punctului de reflexie

    P_reflect_end = P + t_reflect * R

    return (P.tolist(), P_reflect_end.tolist())

def reflect_ray_to_plane2(ray_start, ray_end, triangle_vertices, z_target):
    A, B, C = [np.array(v) for v in triangle_vertices]
    A[2] -= dDD
    B[2] -= dDD
    C[2] -= dDD
    P0 = np.array(ray_start)
    P1 = np.array(ray_end)

    D = P1 - P0
    D = D / np.linalg.norm(D)

    N = np.cross(B - A, C - A)
    N = N / np.linalg.norm(N)

    denom = np.dot(N, D)
    if np.abs(denom) < 1e-6:
        return None  # rază paralelă cu planul

    d = np.dot(N, A)
    t = (d - np.dot(N, P0)) / denom
    if t < 0:
        return None

    P = P0 + t * D

    # def point_in_triangle(P, A, B, C):
    #     v0 = C - A
    #     v1 = B - A
    #     v2 = P - A

    #     dot00 = np.dot(v0, v0)
    #     dot01 = np.dot(v0, v1)
    #     dot02 = np.dot(v0, v2)
    #     dot11 = np.dot(v1, v1)
    #     dot12 = np.dot(v1, v2)

    #     invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
    #     u = (dot11 * dot02 - dot01 * dot12) * invDenom
    #     v = (dot00 * dot12 - dot01 * dot02) * invDenom

    #     return (u >= 0) and (v >= 0) and (u + v <= 1)

    if not point_in_triangle(P, A, B, C):
        return None

    R = D - 2 * np.dot(D, N) * N
    R = R / np.linalg.norm(R)

    if np.abs(R[2]) < 1e-6:
        return None  # rază reflectată paralelă cu planul z=z_target

    t_reflect = (z_target - P[2]) / R[2]
    if t_reflect < 0:
        return None  # intersecția e în spatele punctului de reflexie

    P_reflect_end = P + t_reflect * R

    return (P.tolist(), P_reflect_end.tolist())

def reflect_rays_to_plane(rays, triangles, z_target):
    rays_out = []
    # for tri in triangles:
    #     for line in rays:
    #         ray_out = reflect_ray_to_plane(line[0], line[1], tri, -1.5 * dDD)
    #         if ray_out is not None:
    #             rays_out.append(ray_out)
    for line in rays:
        for tri in triangles:
            ray_out = reflect_ray_to_plane(line[0], line[1], tri, -1.5 * dDD)
            if ray_out is not None:
                rays_out.append(ray_out)
                break
    return rays_out

def reflect_rays_to_plane2(rays, triangles, z_target):
    rays_out = []
    # for tri in triangles:
    #     for line in rays:
    #         ray_out = reflect_ray_to_plane2(line[0], line[1], tri, -1.0 * Bf)
    #         if ray_out is not None:
    #             rays_out.append(ray_out)
    for line in rays:
        for tri in triangles:
            ray_out = reflect_ray_to_plane2(line[0], line[1], tri, -1.0 * Bf)
            if ray_out is not None:
                rays_out.append(ray_out)
                break
    return rays_out

def draw_wireframe_surface(triangles):
    """Desenează triunghiurile primite ca wireframe"""
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    glBegin(GL_TRIANGLES)
    for tri in triangles:
        for i in range(3):
            glVertex3fv(tri[i])
    glEnd()
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

def draw_plane(size):
    
   
    
    #img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img_data = img.tobytes()

    texture_id = glGenTextures(1)
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, texture_id)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.width, img.height, 0,
                 GL_RGB, GL_UNSIGNED_BYTE, img_data)
    



    size = size / 2.0
    glBegin(GL_QUADS)
    glTexCoord2f(0.0, 0.0)
    glVertex3f(-size, -size, 0)
    
    glTexCoord2f(1.0, 0.0)
    glVertex3f( size, -size, 0)
    
    glTexCoord2f(1.0, 1.0)
    glVertex3f( size,  size, 0)
    
    glTexCoord2f(0.0, 1.0)
    glVertex3f(-size,  size, 0)
    glEnd()
    
    glDisable(GL_TEXTURE_2D)

def draw_source_rays(rays):
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glColor4f(0.5, 0.5, 0.5, 0.25)  # gri cu transparență
    
    glBegin(GL_LINES)
    for line in rays:
        glVertex3fv(line[0])
        glVertex3fv(line[1])
    #glVertex3f(0.0, 0.0, 0.0)
    #glVertex3f(0.0, 0.0, -99999.0)
    glEnd()
    glDisable(GL_BLEND)

def draw_intermediate_rays(rays):
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glColor4f(0.75, 0.75, 0.75, 0.15)  # gri cu transparență
    
    glBegin(GL_LINES)
    for line in rays:
        glVertex3fv(line[0])
        glVertex3fv(line[1])
    #glVertex3f(0.0, 0.0, 0.0)
    #glVertex3f(0.0, 0.0, -99999.0)
    glEnd()
    glDisable(GL_BLEND)
    
def draw_final_rays(rays):
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glColor4f(0.95, 0.95, 0.95, 0.35)  # gri cu transparență
    
    glBegin(GL_LINES)
    for line in rays:
        glVertex3fv(line[0])
        glVertex3fv(line[1])
    #glVertex3f(0.0, 0.0, 0.0)
    #glVertex3f(0.0, 0.0, -99999.0)
    glEnd()
    glDisable(GL_BLEND)
    
def draw_text(x, y, text):
    glWindowPos2f(x, y)
    for ch in text:
        glutBitmapCharacter(GLUT_BITMAP_9_BY_15, ord(ch))
        
def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    
    
    glColor3f(1.0, 1.0, 1.0)
    draw_text(10, 90, f"Command keys: h/H, f/F")
    draw_text(10, 70, f"D1 prim mirr: {R:.0f}/{D:.0f}, {d_gura:.0f}")
    draw_text(10, 50, f"D2 seco mirr: {R2:.0f}/{D2:.0f}")
    draw_text(10, 30, f"Dist D1 - D2: {dDD:.1f}")
    draw_text(10, 10, f"Back focus d: {-1*Bf:.1f}")

    gluLookAt(0, 0, zoom, 0, 0, 0, 0, 1, 0)

    glRotatef(rotate_x, 1.0, 0.0, 0.0)
    glRotatef(rotate_y, 0.0, 1.0, 0.0)

    
    glScalef(0.3, 0.3, 0.3)
    

    draw_plane(plane);
    
    glTranslatef(0.0, 0.0, Bf)
    
    glPushMatrix()
    glRotatef(180, 1.0, 0.0, 0.0)
    glTranslatef(0.0, 0.0, -1.0 * Bf)
    #draw_source_rays(source_rays)
    glPopMatrix()
    
    #draw_intermediate_rays(intermediate_rays)
    draw_final_rays(final_rays)
    
    glColor3f(0.7, 0.7, 0.9)
    
    draw_wireframe_surface(surface_triangles)
    glTranslatef(0.0, 0.0, -1 * dDD)
    draw_wireframe_surface(surface_triangles2)
    
       

    glutSwapBuffers()

def reshape(w, h):
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60, w / h, 0.1, 2000.0)
    glMatrixMode(GL_MODELVIEW)

def keyboard(key, x, y):
    global concava, concava2, D, D2, dDD, Bf, surface_triangles, surface2_triangles2

    if key == b'c':
        concava = not concava
        # surface_triangles = generate_spherical_cap_triangles(R, D, N_THETA, M_PHI, concava, d=d_gura)
        # surface_triangles = generate_spherical_cap_triangles_uniform(R, D, N_THETA, M_PHI, concava, d=d_gura)
        # surface_triangles = generate_triangular_grid(R, D, d=d_gura, concave=concava, edge_length=0.1)
        surface_triangles = generate_uniform_triangular_mesh(R, D, d=d_gura, concave=concava, edge_length=l_triunghi)
        glutPostRedisplay()

    elif key == b'+':
        if D + 0.5 < 2 * R:
            D += 0.5
            # surface_triangles = generate_spherical_cap_triangles(R, D, N_THETA, M_PHI, concava, d=d_gura)
            # surface_triangles = generate_spherical_cap_triangles_uniform(R, D, N_THETA, M_PHI, concava, d=d_gura)
            # surface_triangles = generate_triangular_grid(R, D, d=d_gura, concave=concava, edge_length=0.1)
            surface_triangles = generate_uniform_triangular_mesh(R, D, d=d_gura, concave=concava, edge_length=l_triunghi)
            glutPostRedisplay()

    elif key == b'-':
        if D - 0.5 > 0:
            D -= 0.5
            # surface_triangles = generate_spherical_cap_triangles(R, D, N_THETA, M_PHI, concava, d=d_gura)
            # surface_triangles = generate_spherical_cap_triangles_uniform(R, D, N_THETA, M_PHI, concava, d=d_gura)
            # surface_triangles = generate_triangular_grid(R, D, d=d_gura, concave=concava, edge_length=0.1)
            surface_triangles = generate_uniform_triangular_mesh(R, D, d=d_gura, concave=concava, edge_length=l_triunghi)
            glutPostRedisplay()
    elif key == b'H':
        if dDD < R:
            dDD += 0.5
            glutPostRedisplay()
    elif key == b'h':
        if dDD > 0:
            dDD -= 0.5
            glutPostRedisplay()
    elif key == b'F':
        if Bf + 0.5 <= 0:
            Bf += 0.5
            glutPostRedisplay()
    elif key == b'f':
        if Bf > -1*R:
            Bf -= 0.5
            glutPostRedisplay()

def mouse(button, state, x, y):
    global mouse_down, last_x, last_y, zoom

    if button == GLUT_LEFT_BUTTON:
        if state == GLUT_DOWN:
            mouse_down = True
            last_x, last_y = x, y
        else:
            mouse_down = False

    elif button == 3:  # scroll up
        zoom -= 1.0
        if zoom < 2:
            zoom = 2
        glutPostRedisplay()

    elif button == 4:  # scroll down
        zoom += 1.0
        if zoom > 500:
            zoom = 500
        glutPostRedisplay()

def motion(x, y):
    global last_x, last_y, rotate_x, rotate_y, mouse_down

    if mouse_down:
        rotate_x += (y - last_y) * 0.15
        rotate_y += (x - last_x) * 0.15
        last_x, last_y = x, y
        glutPostRedisplay()

def main():
    global surface_triangles, surface_triangles2
    global source_rays, intermediate_rays, final_rays
    global final_ponits
    global img

    glutInit(sys.argv)
    #glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH | GLUT_MULTISAMPLE)
    glutInitWindowSize(900, 700)
    glutCreateWindow(b"RayCassT - Simulator telescop")

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_MULTISAMPLE)
    glEnable(GL_LINE_SMOOTH)
    #glEnable(GL_POINT_SMOOTH)
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
    #glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
    
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    glClearColor(0.1, 0.1, 0.15, 1.0)

    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutMouseFunc(mouse)
    glutMotionFunc(motion)


    #START scenariu simulare

    PARAM = "R2"
    #with open("analiza_puncte.txt", "a") as f:
    #    f.write("Parametru simulat\tValoare parametru\tDispersie\tExcentricitate\tElongatie\n")
    
    # for R2 in np.arange(129.0, 129.0, 1.0):
    #     print(f"Parametru: {PARAM} - {R2:.3f})")
                 
    # surface_triangles = generate_spherical_cap_triangles(R, D, N_THETA, M_PHI, concava, d=d_gura)
    # surface_triangles = generate_spherical_cap_triangles_uniform(R, D, N_THETA, M_PHI, concava, d=d_gura)
    # surface_triangles = generate_triangular_grid(R, D, d=d_gura, concave=concava, edge_length=0.1)
    
    start = time.time()
    surface_triangles = generate_uniform_triangular_mesh(R, D, d=d_gura, concave=concava, edge_length=l_triunghi)
    end = time.time()
    durata = end - start
    print(f"Număr triunghiuri oglinda 1: {len(surface_triangles)} (in {durata:.2f}s)")
    
    start = time.time()
    surface_triangles2 = generate_uniform_triangular_mesh(R2, D2, d=0, concave=concava2, edge_length=l_triunghi2)
    end = time.time()
    durata = end - start
    print(f"Număr triunghiuri oglinda 2: {len(surface_triangles2)} (in {durata:.2f}s)")
    
    start = time.time()
    #source_rays = generate_init_rays(surface_triangles)
    intermediate_rays = generate_init_rays(surface_triangles)
    end = time.time()
    durata = end - start
    # print(f"Număr raze initiale: {len(source_rays)} (in {durata:.2f}s)")
    print(f"Număr raze initiale: {len(intermediate_rays)} (in {durata:.2f}s)")
    
    # start = time.time()
    # intermediate_rays = reflect_rays_to_plane (source_rays, surface_triangles, dDD)
    # end = time.time()
    # durata = end - start
    print(f"Număr raze reflectate 1: {len(intermediate_rays)} (in {durata:.2f}s)")
    
    start = time.time()
    final_rays = reflect_rays_to_plane2 (intermediate_rays, surface_triangles2, dDD)
    end = time.time()
    durata = end - start
    print(f"Număr raze reflectate 2: {len(final_rays)} (in {durata:.2f}s)")
    
    start = time.time()
    img_temp = Image.new('RGB', (int(plane/pixel_size) + 1, int(plane/pixel_size) + 1), color=(0, 0, 0))

    # intersectii = 0
    final_ponits = []
    for line in final_rays:
        p = line[1]
        px = p[0]
        py = p[1]
        if px > -1 * plane / 2.0 and px < plane / 2.0 and py > -1 * plane / 2.0 and py < plane / 2.0:
            color = img_temp.getpixel((int((px + plane / 2.0) / pixel_size), int((py + plane / 2.0) / pixel_size)))[0]
            if color < 256:
                color += 1;
            img_temp.putpixel((int((px + plane / 2.0) / pixel_size), int((py + plane / 2.0) / pixel_size)), (color, color, color))
            final_ponits.append(p)
            # intersectii += 1


    # 1. Centroid
    centroid = np.mean(final_ponits, axis=0)
    
    # 2. Dispersie (media pătratică a distanțelor față de centroid)
    distances = np.linalg.norm(final_ponits - centroid, axis=1)
    mean_squared_distance = np.mean(distances**2)
    
    # 3. PCA pentru excentricitate și elongare
    pca = PCA(n_components=2)
    pca.fit(final_ponits)
    explained_variance = pca.explained_variance_
    
    # Excentricitate = sqrt(1 - (b^2 / a^2)) pentru elipsa echivalentă
    a2, b2 = explained_variance
    excentricity = np.sqrt(1 - b2 / a2)
    elongation = np.sqrt(a2 / b2)
    
    axis_lengths = 2 * np.sqrt(pca.explained_variance_)  # diametrul axelor
    axis_major = axis_lengths[0]
    axis_minor = axis_lengths[1]

    # Afișare rezultate
    print(f"Centroid: {centroid}")
    print(f"Dispersie (media pătratică a distanțelor): {mean_squared_distance:.3f}")
    print(f"Excentricitate: {excentricity:.3f} (0-cerc, 1-alungit)")
    print(f"Axa mare / axa mică): {axis_major:.1f} / {axis_minor:.1f}")
    print(f"Elongatie (axa mare / axa mică): {elongation:.3f}")

    # points = np.array([...])  # punctele tale 2D, n x 2
    points = np.array(final_ponits)
    min_x = np.min(points[:, 0])
    max_x = np.max(points[:, 0])
    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])
    
    # Bounding box = colțul stânga-jos + dimensiuni
    width = max_x - min_x
    height = max_y - min_y
    print(f"Bounding box-ul): {width:.1f} x {height:.1f}")

    #with open("analiza_puncte.txt", "a") as f:
    #    #f.write("Param\tParam value\tDispersie\tExcentricitate\tElongare\n")
    #    f.write(f"{PARAM}\t{R2:.3f}\t{mean_squared_distance:.3f}\t{excentricity:.3f}\t{elongation:.3f}\n")
    
    draw = ImageDraw.Draw(img_temp)
    draw.rectangle([int((min_x + plane / 2.0) / pixel_size), int((min_y + plane / 2.0) / pixel_size),  int((max_x + plane / 2.0) / pixel_size), int((max_y + plane / 2.0) / pixel_size)], outline="red", width=1)
    
    filename = f"spot {PARAM} {R2:.3f}.png"
    ImageOps.autocontrast(img_temp).save(filename)
    img = ImageOps.autocontrast(img_temp.resize((256, 256), Image.LANCZOS))
    # img.save("spot.png")
    end = time.time()
    durata = end - start
    print(f"Număr raze  pe  senzor: {len(final_ponits)} (in {durata:.2f}s)")
    
    glutMainLoop()

if __name__ == "__main__":
    main()
