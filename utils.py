# Copyright Martin Lueker-Boden
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import yaml
import numpy as np
import numpy.linalg
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.signal as signal
import math

class SupportPoint:
    def __init__(self, heatmap, x, y, id):
        '''heatmap: a HeatMap Object'''
        self.map   = heatmap
        self.x     = x
        self.y     = y
        self.above = None
        self.below = None
        self.left  = None
        self.right = None
        self.id    = id

    def propagate_id(self):
        '''Depth-first search algorithm to assign all connected pixels the same
        id. At the end of the algorithm, all connected points will have the ID
        of the lowest connected neighbor'''
        isolated = True
        neighborhood = []
        self.map.stack_depth += 1
        for neighbor in [self.above, self.below, self.left, self.right]:
            if (neighbor != None):
                isolated = False
                if(neighbor.id > self.id):
                    neighbor.id = self.id
                    unclaimed_neighbors = neighbor.propagate_id()
                    if (unclaimed_neighbors != None):
                        neighborhood.extend(unclaimed_neighbors)
                    neighborhood.append(neighbor)
        self.map.stack_depth -= 1
        if ( (len(neighborhood) > 0) or isolated):
            return neighborhood
        else:
            return None

class Contact:
    MaxByte = 256
    # Lookup table for quickly computing logarithms.
    ln_lut = np.log(np.arange(MaxByte));

    def __init__(self, map, points, id):
        self.map = map
        self.data = np.zeros([self.map.height, self.map.width])
        self.points = points
        self.id = id
        self.sign = 0
        self.generate_map()
        self.calc_sign()
        self.calc_params()

    def generate_map(self):
        self.npoints = len(self.points)
        self.x = np.zeros(self.npoints, np.int32)
        self.y = np.zeros(self.npoints, np.int32)
        self.z = np.zeros(self.npoints)
        self.curvature = np.zeros(self.npoints)
        self.hess_xx = np.zeros(self.npoints)
        self.hess_yy = np.zeros(self.npoints)
        self.hess_xy = np.zeros(self.npoints)
        for i in range(0, self.npoints):
            p = self.points[i]
            self.x[i] = p.x
            self.y[i] = p.y
            self.z[i] = self.map.data[p.y, p.x]
            self.curvature[i] = self.map.curvature[p.y, p.x]
            self.hess_xx[i] = self.map.hess_xx[p.y, p.x]
            self.hess_yy[i] = self.map.hess_yy[p.y, p.x]
            self.hess_xy[i] = self.map.hess_xy[p.y, p.x]
            # TODO: Remove this
            self.data[p.y, p.x] = self.map.data[p.y, p.x]

    def calc_sign(self):
        hm_max = np.amax(self.z)
        hm_min = np.amin(self.z)
        if (abs(hm_max) > abs(hm_min)):
            self.sign = 1
        else:
            self.sign = -1

    # TODO: Try using CBLAS routines for ease of porting vector ops to
    #       C.  When this is done the port C-code libCBLAS will allow for
    #       optimized libraries to make use of CPU vector extensions (e.g. AVX)
    def calc_moments(self):
        # For this routine consider all values as positive
        # Real contacts and inverted "ghost" contacts are swapped as
        # needed to have a mostly positive sign.
        #
        # Ignore zero values or values whose sign differs from most
        # of the other points
        z_unsigned  = self.z * self.sign
        actpts = np.asarray(z_unsigned > 0)
        x  = self.x[actpts]
        y  = self.y[actpts]
        z  = z_unsigned[actpts]

        # various products over x & y, one value per support point
        x2    = np.power(x, 2)
        x3    = np.power(x, 3)
        x4    = np.power(x, 4)
        y2    = np.power(y, 2)
        y3    = np.power(y, 3)
        y4    = np.power(y, 4)
        xy    = x * y
        x2y   = x2 * y
        xy2   = x * y2
        x2y2  = x2 * y2

        # weighting values: z^2 and z2 * log(z)
        z2    = np.power(z, 2);
        z2lnz = z2 * Contact.ln_lut[np.asarray(z, np.int8)]

        # Compute the moments here, taking
        # dot-products of the vectors above.
        moments = {};
        moments["z2"]       = np.sum(z2)
        moments["z2x"]      = z2 @ x
        moments["z2x2"]     = z2 @ x2
        moments["z2x3"]     = z2 @ x3
        moments["z2x4"]     = z2 @ x4
        moments["z2y"]      = z2 @ y
        moments["z2y2"]     = z2 @ y2
        moments["z2y3"]     = z2 @ y3
        moments["z2y4"]     = z2 @ y4
        moments["z2xy"]     = z2 @ xy
        moments["z2xy2"]    = z2 @ xy2
        moments["z2x2y"]    = z2 @ x2y
        moments["z2x2y2"]   = z2 @ x2y2
        moments["z2lnz"]    = np.sum(z2lnz)
        moments["z2lnzx"]   = z2lnz @ x
        moments["z2lnzx2"]  = z2lnz @ x2
        moments["z2lnzy"]   = z2lnz @ y
        moments["z2lnzy2"]  = z2lnz @ y2
        return moments

    def calc_params(self):
        moments = self.calc_moments()

        mat = np.zeros([5, 5])
        mat[0, 0]                         = moments["z2"]
        mat[1, 0] = mat[0, 1]             = moments["z2x"]
        mat[2, 0] = mat[1, 1] = mat[0, 2] = moments["z2x2"]
        mat[3, 0] = mat[0, 3]             = moments["z2y"]
        mat[4, 0] = mat[3, 3] = mat[0, 4] = moments["z2y2"]
        mat[2, 1] = mat[1, 2]             = moments["z2x3"]
        mat[3, 1] = mat[1, 3]             = moments["z2xy"]
        mat[4, 1] = mat[1, 4]             = moments["z2xy2"]
        mat[2, 2]                         = moments["z2x4"]
        mat[3, 2] = mat[2, 3]             = moments["z2x2y"]
        mat[4, 2] = mat[2, 4]             = moments["z2x2y2"]
        mat[4, 3] = mat[3, 4]             = moments["z2y3"]
        mat[4, 4]                         = moments["z2y4"]

        vec = np.ndarray((5,))
        vec[0] = moments["z2lnz"]
        vec[1] = moments["z2lnzx"]
        vec[2] = moments["z2lnzx2"]
        vec[3] = moments["z2lnzy"]
        vec[4] = moments["z2lnzy2"]

        # TODO: Exception handling
        try:
            mat_inv = numpy.linalg.inv(mat)
            self.params = mat_inv @ vec
        except numpy.linalg.LinAlgError:
            print(mat)
            self.params = []

    def model_val(self, x, y):
        poly =  self.params[0] + self.params[1] * x + self.params[2] * x * x;
        poly += self.params[3] * y + self.params[4] * y * y;
        return math.exp(poly) * self.sign

    # Add data from this contact to a combined map
    def update_map(self, data, type="id"):
        for i in range(0, self.npoints):
            if (type == "id"):
                data[self.y[i], self.x[i]] = self.id + 1
            elif (type == "val"):
                data[self.y[i], self.x[i]] = self.z[i]
            elif (type == "model"):
                data[self.y[i], self.x[i]] = self.model_val(self.x[i],
                                                            self.y[i])
            elif (type == "partition_p"):
                data[self.y[i], self.x[i]] = self.partition_p[i]
            elif (type == "partition_m"):
                data[self.y[i], self.x[i]] = self.partition_m[i]
        return data

    def calc_partition(self):
        idx = np.argmin(self.curvature)
        curv = self.curvature[idx]
        hess_sum = self.hess_xx[idx] + self.hess_yy[idx]
        discriminant = math.sqrt(hess_sum - 4 * curv)
        lambda_p = (hess_sum + discriminant)/2
        lambda_m = (hess_sum - discriminant)/2
        # TODO: citation to eigenvector calculation
        normal_vp = (self.hess_xy[idx], lambda_p - self.hess_xx[idx])
        normal_vm = (self.hess_xy[idx], lambda_m - self.hess_xx[idx])
        offset_p = normal_vp[0] * self.x[idx] + normal_vp[1] * self.y[idx]
        offset_m = normal_vm[0] * self.x[idx] + normal_vm[1] * self.y[idx]
        self.partition_p = np.zeros(self.npoints)
        self.partition_m = np.zeros(self.npoints)
        for i in range(0, self.npoints):
            d_p = normal_vp[0] * self.x[i] + normal_vp[1] * self.y[i]
            d_m = normal_vm[0] * self.x[i] + normal_vm[1] * self.y[i]
            self.partition_p[i] = d_p - offset_p
            self.partition_m[i] = d_m - offset_m

    # Returns True if the partition has regions of positive curvature on
    # both sides
    def check_partition(self):
        pos_found = False
        neg_found = False
        for i in range(0, self.npoints):
            if (self.partition_m[i] < 0):
                neg_found = neg_found or (self.curvature[i] > 0)
            if (self.partition_m[i] > 0):
                pos_found = pos_found or (self.curvature[i] > 0)
        return neg_found and pos_found

class HeatMap:
    def __init__(self, d):
        '''Takes in a dictionary with keys "height", "width" and "data"'''
        self.width = d["width"]
        self.height = d["height"]
        self.data = np.reshape(np.array(d["data"]), [self.height, self.width])
        mode = stats.mode(self.data, None)
        # Contact heatmaps can create negative swings
        # So the zero cannot be identified by taking the minimum
        self.data -= mode[0]
        self.calc_curvature()
        self.identify_supports()
        self.link_supports()
        self.identify_contacts()

    # TODO: accept_surrounded does not seem to be working
    def is_support_point(self, x, y, accept_surrounded=True):
        '''Returns true if point (x, y) is non-zero, (or is surrounded by non
           zero points)'''
        if (self.data[y, x] != 0):
            return True
        if (not accept_surrounded):
            return False

        # To determine if a pixel is surrounded by non-zero points look
        # at the values of the neigboring points.
        score = 0
        score += (x != 0                and self.data[y, x-1] != 0)
        score += (x < (self.width - 1)  and self.data[y, x+1] != 0)
        score += (y != 0                and self.data[y-1, x] != 0)
        score += (y < (self.height -1)  and self.data[y+1, x] != 0)
        return (score == 4)

    def identify_supports(self):
        id = 0
        self.supports = {}
        self.support_list = []
        for i in range(0, self.width):
            for j in range(0, self.height):
                if self.is_support_point(i, j):
                    point = SupportPoint(self, i, j, id)
                    self.supports[(j, i)] = point
                    self.support_list.append(point)
                    id += 1

    def link_supports(self):
        for s in self.support_list:
        # link with support points above and to the left
        # Note: This
            i = s.x
            j = s.y
            if (i != 0 and (j, i - 1) in self.supports):
                neighbor = self.supports[(j, i - 1)]
                neighbor.left = s
                s.right = neighbor
            if (j != 0 and (j - 1, i) in self.supports):
                neighbor = self.supports[(j - 1, i)]
                s.above = neighbor
                neighbor.below = s

    def identify_contacts(self):
        self.stack_depth = 0
        current_id = -1
        self.contacts = []
        #
        # Break the overall support list into
        #
        # Support points are assigned ids based on the order they are found
        #
        # By also propagating support point ids in the order they are found,
        # we can be sure that each partition of the support graph has
        # the id of the first point found.
        contact_id = 0
        for s in self.support_list:
            neighborhood = s.propagate_id()
            if(neighborhood != None):
                neighborhood.append(s)
                c = Contact(self, neighborhood, contact_id)
                if (len(c.params) == 0):
                    print("Failed to analyze contact {}".format(contact_id))
                self.contacts.append(c)
                contact_id += 1;

    def calc_curvature(self):
        sobel_xx = np.asarray([[1, -2, 1], [2, -4, 2], [1, -2, 1]], np.float32)
        sobel_yy = np.asarray([[1, 2, 1], [-2, -4, -2], [1, 2, 1]], np.float32)
        sobel_xy = np.asarray([[1, 0, -1], [0, 0, 0], [-1, 0, 1]], np.float32)

        self.hess_xx = signal.convolve2d(self.data, sobel_xx, mode="same")
        self.hess_yy = signal.convolve2d(self.data, sobel_yy, mode="same")
        self.hess_xy = signal.convolve2d(self.data, sobel_xy, mode="same")

        # The scalar (Gaussian) curvature is the determinant of the
        # hessian matrix
        self.curvature    = self.hess_xx * self.hess_yy - self.hess_xy**2
        self.hess_sum     = self.hess_xx + self.hess_yy;
        self.discriminant = np.sqrt(self.hess_sum**2 - 4 * self.curvature);
        self.lambda_m     = (self.hess_sum - self.discriminant)/2
        self.lambda_p     = (self.hess_sum + self.discriminant)/2


    def plot(self, mode = "data"):
        fig, ax = plt.subplots()
        if (mode == "data"):
            map = self.data
        elif (mode == "support_idx"):
            map = np.zeros([self.height, self.width])
            for k, v in self.supports.items():
                map[v.y, v.x] = v.id
        elif (mode == "contact_idx"):
            map = np.zeros([self.height, self.width])
            for c in self.contacts:
                map = c.update_map(map, type="id")
        elif (mode == "model"):
            map = np.zeros([self.height, self.width])
            for c in self.contacts:
                map = c.update_map(map, type="model")
        elif (mode == "curvature"):
            map = self.curvature
        elif (mode == "curvxdata"):
            map = self.curvature * self.data
        elif (mode == "partition_p"):
            map = np.zeros([self.height, self.width])
            for c in self.contacts:
                map = c.update_map(map, type="partition_p")
        elif (mode == "partition_m"):
            map = np.zeros([self.height, self.width])
            for c in self.contacts:
                map = c.update_map(map, type="partition_m")
        im_min = np.amin(map)
        im_max = np.amax(map)
        im_range = max(abs(im_min), abs(im_max))

        im = ax.imshow(map, vmin = -im_range, vmax = im_range,
                       cmap = plt.get_cmap('seismic'))

        plt.show()

def load_yaml_data(filepath, RawOnly=False):
    result = []
    with open(filepath) as f:
        raw=yaml.load(f, Loader=yaml.CSafeLoader)
    if (RawOnly):
        return None, raw
    for i in range(len(raw["heatmaps"])):
        d = raw["heatmaps"][i]
        try:
            hm = HeatMap(d)
        except Exception:
            print("Failed to completely analyse frame {}".format(i))
        result.append(hm)
    return result, raw

def play_heatmaps(heatmaps, frames=(-1,-1), delta_t=0.1):
    fig, ax = plt.subplots()

    hm_min = np.amin(heatmaps[0].data)
    hm_max = np.amax(heatmaps[0].data)

    if (frames[0] < 0):
        frame_range = range(0, len(heatmaps))
    else:
        frame_range = range(frames[0], frames[1])

    for i in frame_range:
        hm_min = min(hm_min, np.amin(heatmaps[i].data));
        hm_max = max(hm_min, np.amax(heatmaps[i].data));

    hm_range = max(abs(hm_min), abs(hm_max))

    for i in frame_range:
        ax.cla()
        ax.imshow(heatmaps[i].data, vmin=-hm_range, vmax=hm_range,
                  cmap = plt.get_cmap('seismic'))
        ax.set_title("frame {}".format(i))
        plt.pause(delta_t)
