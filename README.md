# ShapeDescriptor

a python realization of the paper:

	Description of 3D-shape using a complex function on the sphere

	D.V. Vranic, D. Saupe, 2002

Note: The previous dependency on python library trimesn has been removed. Currently you can use numpy array to get the descriptor like this

	from descriptor import descriptorCS
	
	shape_descriptor = descriptorCS(vertices, facet)
	
where vertices is (n,3) numpy array, facet is (m,3) numpy array.(n = # vertices, m = # facet) 
  
Function:(see descriptor.py)

  descriptorRS(vertices,facet, coef_num_sqrt=13):
  
    apply spherical harmonics transform on the real function on the sphere S^2
    
    and use the truncated coefficients as the descriptor of the given mesh
    
  descriptorSS(vertices,facet, coef_num_sqrt=13):
  
    apply spherical harmonics transform on the imaginary part of complex function on the sphere S^2
    
    and use the truncatedcoefficients as the descriptor of the given mesh
    
  descriptorCS(vertices,facet, coef_num_sqrt=13):
  
    apply spherical harmonics transform on the complex function on the sphere S^2
    
    and use the truncated coefficients as the descriptor of the given mesh
