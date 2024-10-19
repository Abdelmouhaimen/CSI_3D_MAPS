import obja
import numpy as np
import sys
import queue
from scipy.special import Delaunay
import shapely
from shapely.geometry import Polygon, MultiPoint, LineString
from shapely.ops import triangulate


class Decimater(obja.Model):
    """
    
    """
    def __init__(self):
        super().__init__()
        self.deleted_faces =  queue.Queue()
        self.neighbor_faces = {}
        self.neighbor_vertices = {}
        self.deleted_vertices = []


    '''Trouver les faces voisines de tous les sommets et les stocker dans un dictionnaire'''
    def find_1ring_neighborhood(self) :
        for  (vertex_index, vertex) in enumerate(self.vertices):
            self.neighbor_faces [vertex_index] =  []
            self.neighbor_vertices[vertex_index] = set()
            for (face_index, face) in enumerate(self.faces):
                if  vertex_index in  [face.a,face.b,face.c] :
                    self.neighbor_faces[vertex_index].append(face_index)
                    # Trouver les indices des sommets voisins dans le 1-ring. 
                    ind = set([face.a,face.b,face.c].remove(vertex_index))
                    self.neighbor_vertices[vertex_index] = self.neighbor_vertices[vertex_index].union(ind)


    '''Calculer la normale en chaque point de notre maillage 3D 
    comme la moyenne pondérée des normales des surfaces voisines
    '''
    def compute_normal_vertex(self, vertex_index):
        neighbor_faces  = self.neighbor_faces[vertex_index]
        n = 0
        A = 0
        for face_index in neighbor_faces :
            ni,ai =self.normal_surface(self ,face_index )
            n += ai*ni
            A += ai
        return n/A
    
    ''' Calculer la normale de chaque face voisine de notre point '''
    def normal_surface(self ,face_index ) :
        face = self.faces[face_index]
        v1 = self.vertices[face.a].T - self.vertices[face.b].T
        v2 = self.vertices[face.a].T - self.vertices[face.c].T
        ni = np.cross(v1,v2)
        ai = np.linalg.norm(ni)
        return ni,ai



    def get_tangent_vectors(self,n):
        if abs(n[0]) > abs(n[1]):
            u = np.array([-n[2], 0, n[0]]) / np.sqrt(n[0]**2 + n[2]**2)
        else:
            u = np.array([0, n[2], -n[1]]) / np.sqrt(n[1]**2 + n[2]**2)
        v = np.cross(n, u)
        return u, v
    


    ''' cCalculer la courbure en un point en estimant la surface par la méthode des moindres carrés'''
    def calculate_curvature(self, vertex_index) :

        ni = self.compute_normal_vertex( vertex_index)
        u, v = self.get_tangent_vectors(ni)
        
        # Préparer les matrices pour la minimisation des moindres carrés
        A = []
        b = []
        neighbor_vertex = self.neighbor_vertices[vertex_index]
    
        for j in neighbor_vertex:
            Pj = self.vertices[j].T
            Pi = self.vertices[vertex_index].T
            P_diff = Pj - Pi
            uj = np.dot(P_diff, u)
            vj = np.dot(P_diff, v)
            wj = np.dot(P_diff, ni)
            
            
            A.append([uj**2, 2*uj*vj, vj**2])
            b.append(wj)
        
        
        A = np.array(A)
        b = np.array(b)
        
        
        x = np.linalg.lstsq(A, b, rcond=None)[0]
        a, b, c = x
        
        
        H = a + c  
        K = a * c - b**2  
        
        
        discriminant = np.sqrt(H**2 - K)
        k1 = H + discriminant
        k2 = H - discriminant
        k =abs (k1) + abs(k2)
    
        return k
    

    '''for a vertex p_i ∈P_l, we consider its 1-ring neighborhood φ(|star(i)|) and compute its area a(i)   '''
    def calculate_area(self , vertex_index):
        faces  = self.neighbor_faces[vertex_index]
        A = 0
        for face_index in faces :
            _,ai = self.normal_surface(face_index)
            A += ai/2
        return A 
    

    def calculate_priority(self,lambd =1/2) :
        
        A = []
        K = []
        max_A = -1
        max_K = -1
        for  (vertex_index, vertex) in enumerate(self.vertices):
            ring_1=list(self.neighbor_vertices[vertex_index])
            ai = self.calculate_area(vertex_index)
            ki = self.calculate_curvature(vertex_index)
            if len(ring_1) < 12 :
                A.append(ai)
                K.append(ki)

            if max_A <= ai :
                max_A = ai
            if max_K <= ki :
                max_K = ki
        w = list((lambd/max_A) * np.array(A)  + ((1 - lambd)/ max_K)* np.array(K))
        dict = {i:l for i,l in enumerate(w)}
        indices = sorted(dict , reverse = True)
        unremovable = [];
        for i in indices:
            if i not in unremovable :
                self.deleted_vertices.put(i)
                unremovable.extend(list(self.neighbor_vertices[i]))
    

    """
    Organise les sommets voisins d'un sommet donné dans un ordre cyclique
    en suivant les arêtes dans le maillage.
    """
    def find_cyclique(self,vertex_index) :
        """
        Crée un dictionnaire d'arêtes reliant les sommets dans le voisinage d'un sommet central.
        Les arêtes sont orientées de manière à permettre un parcours cyclique.
        """
        def get_edges(adjacent_faces) :
            edges = {}
            for face_index in adjacent_faces:
                face = self.faces[face_index]
                vertices_in_face = [face.a, face.b, face.c]
                vertices_in_face.remove(vertex_index)
                edges[vertices_in_face[0]] = vertices_in_face[1]
                edges[vertices_in_face[1]] = vertices_in_face[0]
            return edges
        
        ring_1  = self.neighbor_vertices[vertex_index]
        faces = self.neighbor_faces[vertex_index]
        edges = get_edges(faces)
        vertex_0 =  ring_1[0]
        cyclique_ring_1 = [vertex_0]
        current_vertex = vertex_0
        while len(cyclique_ring_1) < len(ring_1) :
            next_vertex = edges[current_vertex] 
            cyclique_ring_1.append(next_vertex)
            current_vertex  = current_vertex
        return cyclique_ring_1.append(vertex_0)

    """
    Applique une carte conforme pour projeter les sommets voisins dans un plan.
    """
    def remove_vertex_and_retriangulate(self, vertex_index) :
        cycle = self.find_cyclique(self,vertex_index)
        neighbors = self.vertices[cycle,:]
        diff = neighbors - self.vertices[vertex_index]
        r  = np.linalg.norm(diff , axis= 1, keepdims=True)
        produits_scalaires = np.sum(diff[:-1] * diff[1:], axis=1)
        cos_theta = produits_scalaires / (np.linalg.norm(diff[1:] ,axis= 1 )) * (np.linalg.norm(diff[:-1] ,axis= 1 ))
        theta = np.expand_dims(np.arccos(np.clip(cos_theta ,-1,1)),axis=1)
        upper = np.tril(np.ones((theta.shape[0],theta.shape[0])))
        total_angle = upper @ theta 
        positions_2D = np.zeros((len(self.neighbor_vertices[vertex_index]), 2))
        positions_2D[:,0] = r * np.cos(total_angle)  
        positions_2D[:,1] = r * np.sin(total_angle)  
   






        

        





        
        


                
            







        

        
        

        
        



                


            



            









    
    
                    








