"""
@author: rscalia
@date: Mon 29/06/2020

Questo componente implementa l'Algoritmo di Tracking dei Centroidi.

"""

from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np


class CentroidTracker():

	def __init__(self, pMaxDisappeared=50):
		
		self._nextObjectID 		= 1
		self._objects 			= OrderedDict()
		self._disappeared 		= OrderedDict()
		self._trackedObjects 	= OrderedDict()
		self._maxDisappeared 	= pMaxDisappeared


	def register(self, pCentroid, pBbox):
		
		self._objects					[self._nextObjectID] 			= pCentroid
		self._trackedObjects 			[self._nextObjectID] 			= pBbox
		self._disappeared				[self._nextObjectID] 			= 0
		self._nextObjectID 												+= 1


	def deregister(self, pObjectID):
		del self._objects				[pObjectID]
		del self._disappeared			[pObjectID]
		del self._trackedObjects		[pObjectID]


	def update(self, pRects):
		
		if len(pRects) == 0:
			
			for object_ID in list(self._disappeared.keys()):
				self._disappeared[object_ID] += 1

				
				if self._disappeared[object_ID] > self._maxDisappeared:
					self.deregister(object_ID)

			return self._trackedObjects

		input_centroids = np.zeros((len(pRects), 2), dtype="int")

		
		for (i, (start_x, start_y, end_x, end_y)) in enumerate(pRects):
			c_x = int((start_x + end_x) / 2.0)
			c_y = int((start_y + end_y) / 2.0)
			input_centroids[i] = (c_x, c_y)

		
		if len(self._objects) == 0:
			for i in range(0, len(input_centroids)):
				self.register(input_centroids[i], pRects[i])

		else:
			object_IDs = list(self._objects.keys())
			object_centroids = list(self._objects.values())

			D = dist.cdist(np.array(object_centroids), input_centroids)

			rows = D.min(axis=1).argsort()
			cols = D.argmin(axis=1)[rows]
			used_rows = set()
			used_cols = set()

		
			for (row, col) in zip(rows, cols):
	
				if row in used_rows or col in used_cols:
					continue

				object_ID 										= 		object_IDs[row]
				self._objects			[object_ID] 			=	 	input_centroids[col]
				self._trackedObjects	[object_ID]   			= 		pRects[col]
				self._disappeared		[object_ID] 			= 		0

				
				used_rows.add(row)
				used_cols.add(col)


			unused_rows = set(range(0, D.shape[0])).difference(used_rows)
			unused_cols = set(range(0, D.shape[1])).difference(used_cols)

			if D.shape[0] >= D.shape[1]:

				for row in unused_rows:
					
					object_ID 					 = object_IDs[row]
					self._disappeared[object_ID] += 1

					if self._disappeared[object_ID] > self._maxDisappeared:
						self.deregister(object_ID)

			else:
				for col in unused_cols:
					self.register(input_centroids[col], pRects[col])

		return self._trackedObjects